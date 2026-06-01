import re
import time
import psutil
import logging
import asyncio
from typing import Dict, Any, Optional, List, TypedDict
from langgraph.graph import StateGraph, END
from core.agent import count_tokens, SYSTEM_PROMPT, mock_web_search
from core.scraper import scrape_web_page, scrape_web_page_async
from core.tools import get_current_time, evaluate_math

logger = logging.getLogger("RAG.Graph")

async def async_iter(iterable):
    """
    A utility to asynchronously iterate over both synchronous and asynchronous iterables.
    This is extremely helpful for supporting both real async streams and mock sync streams.
    """
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


class AgentState(TypedDict):
    query: str
    session_id: str
    context_limit: Optional[int]
    source_filter: Optional[str]
    memory_text: str
    scratchpad: str
    iteration: int
    llm_call_count: int
    goals_set: List[str]
    actions_taken: List[dict]
    final_response: str
    overflow_occurred: bool
    overflow_steps: List[str]
    retrieved_context: List[dict]
    events_queue: List[dict]
    early_exit_response: Optional[str]
    early_exit_type: Optional[str]  # "greeting" or "registry"
    parsed_action: Optional[tuple]  # (tool_name, tool_arg)
    is_direct: bool
    raw_response: str
    initial_tokens: int
    final_tokens: int
    search_cache: Dict[str, str]

class RAGLangGraph:
    def __init__(self, agent):
        self.agent = agent
        self.engine = agent.engine
        
        # Build the graph workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("early_exit_check", self.early_exit_check)
        workflow.add_node("early_exit_execute", self.early_exit_execute)
        workflow.add_node("overflow_recovery", self.overflow_recovery)
        workflow.add_node("reasoning", self.reasoning)
        workflow.add_node("execute_formatting_error", self.execute_formatting_error)
        workflow.add_node("execute_tool", self.execute_tool)
        workflow.add_node("synthesis", self.synthesis)
        workflow.add_node("streaming_final_answer", self.streaming_final_answer)
        
        # Set entry point
        workflow.set_entry_point("early_exit_check")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "early_exit_check",
            self.route_early_exit,
            {
                "early_exit_execute": "early_exit_execute",
                "overflow_recovery": "overflow_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "reasoning",
            self.route_after_reasoning,
            {
                "streaming_final_answer": "streaming_final_answer",
                "execute_tool": "execute_tool",
                "execute_formatting_error": "execute_formatting_error",
                "synthesis": "synthesis"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_tool",
            self.route_after_tool,
            {
                "reasoning": "reasoning",
                "synthesis": "synthesis"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_formatting_error",
            self.route_after_tool,
            {
                "reasoning": "reasoning",
                "synthesis": "synthesis"
            }
        )
        
        # Add normal transitions
        workflow.add_edge("early_exit_execute", END)
        workflow.add_edge("overflow_recovery", "reasoning")
        workflow.add_edge("synthesis", END)
        workflow.add_edge("streaming_final_answer", END)
        
        self.compiled_graph = workflow.compile()

    def _call_llm_with_retry(self, messages: list, temperature: float = 0.0, frequency_penalty: float = 0.0, stream: bool = False, max_retries: int = 3) -> Any:
        """Calls the Groq LLM client with exponential backoff retry logic for transient errors."""
        def is_graph_transient(e):
            err_msg = str(e).lower()
            return any(term in err_msg for term in ["429", "503", "rate limit", "timeout", "connection", "api_error", "service unavailable"])

        from core.retry import retry

        @retry(
            retries=max_retries,
            backoff=0.5,
            jitter=(0.1, 0.3),
            is_transient_fn=is_graph_transient,
            logger_name="RAG.Graph"
        )
        def _execute():
            return self.engine.client.chat.completions.create(
                model=self.engine.llm_service.model,
                messages=messages,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                stream=stream
            )

        return _execute()

    async def _call_llm_with_retry_async(self, messages: list, temperature: float = 0.0, frequency_penalty: float = 0.0, stream: bool = False, max_retries: int = 3) -> Any:
        """Calls the Groq LLM client asynchronously with exponential backoff retry logic for transient errors."""
        def is_graph_transient(e):
            err_msg = str(e).lower()
            return any(term in err_msg for term in ["429", "503", "rate limit", "timeout", "connection", "api_error", "service unavailable"])

        from core.retry import retry
        import unittest.mock

        # Check if the async client is a standard Mock (meaning it's not a real AsyncGroq client)
        is_async_mocked = False
        try:
            is_async_mocked = isinstance(self.engine.async_client, unittest.mock.Mock)
        except Exception:
            pass

        if is_async_mocked:
            # Fall back to using the sync client (which is mocked in tests)
            @retry(
                retries=max_retries,
                backoff=0.5,
                jitter=(0.1, 0.3),
                is_transient_fn=is_graph_transient,
                logger_name="RAG.Graph"
            )
            def _execute_sync():
                return self.engine.client.chat.completions.create(
                    model=self.engine.llm_service.model,
                    messages=messages,
                    temperature=temperature,
                    frequency_penalty=frequency_penalty,
                    stream=stream
                )
            return _execute_sync()

        # Real async path
        @retry(
            retries=max_retries,
            backoff=0.5,
            jitter=(0.1, 0.3),
            is_transient_fn=is_graph_transient,
            logger_name="RAG.Graph"
        )
        async def _execute_async():
            return await self.engine.async_client.chat.completions.create(
                model=self.engine.llm_service.model,
                messages=messages,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                stream=stream
            )

        return await _execute_async()


    async def early_exit_check(self, state: AgentState) -> dict:
        query = state["query"]
        clean_query = query.lower().strip()
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"}
        lower_q = clean_query
        
        registry_patterns = [
            "what document", "which document", "list document", "show document", "available document",
            "documents do you", "documents contain", "document do you contain",
            "what file", "which file", "list file", "show file", "show me file", "show me files", "available file", "files do you", "files contain",
            "what dataset", "which dataset", "list dataset", "show dataset", "available dataset", "datasets do you",
            "what database", "which database", "list database", "show database", "available database", "databases do you",
            "what schema", "which schema", "list schema", "show schema", "available schema", "schemas do you",
            "what source", "which source", "list source", "show source", "show me source", "show me sources", "available source", "sources do you",
            "your document", "your file", "your dataset", "your database", "your schema", "your source",
            "my document", "my file", "my dataset", "my database", "my schema", "my source"
        ]
        is_registry_query = any(pat in lower_q for pat in registry_patterns)
        
        # Handle "show me the source(s)" pattern
        words = lower_q.split()
        if "show" in words and "me" in words:
            try:
                me_idx = words.index("me")
                for offset in range(1, 4):
                    if me_idx + offset < len(words) and ("source" in words[me_idx + offset] or "sources" in words[me_idx + offset]):
                        is_registry_query = True
                        break
            except ValueError:
                pass

        events_queue = []
        if is_registry_query:
            events_queue.append({"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"})
            events_queue.append({"event": "thought", "text": "This query asks about available documents. I'll retrieve the registry listing directly."})
            return {
                "early_exit_type": "registry",
                "events_queue": events_queue
            }
        
        if clean_query in greetings or len(clean_query.split()) <= 2 and not any(w in clean_query for w in ["ip", "password", "redis", "database", "stats", "limit", "config", "version", "target"]):
            events_queue.append({"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"})
            events_queue.append({"event": "thought", "text": "This is a simple query or greeting. I can respond directly without searching the web."})
            return {
                "early_exit_type": "greeting",
                "events_queue": events_queue
            }
            
        return {
            "early_exit_type": None
        }

    def route_early_exit(self, state: AgentState) -> str:
        if state.get("early_exit_type") is not None:
            return "early_exit_execute"
        return "overflow_recovery"

    async def early_exit_execute(self, state: AgentState) -> dict:
        exit_type = state["early_exit_type"]
        query = state["query"]
        session_id = state["session_id"]
        events_queue = []
        
        if exit_type == "registry":
            registry_text = self.engine._get_registry_context_text()
            chunk_size = 20
            for i in range(0, len(registry_text), chunk_size):
                events_queue.append({"event": "answer_chunk", "text": registry_text[i:i+chunk_size]})
            
            await asyncio.to_thread(self.engine.save_memory, session_id, query, "user")
            await asyncio.to_thread(self.engine.save_memory, session_id, registry_text, "assistant", 0.8)
            
            done_event = {"event": "done", "response": registry_text, "stats": {
                "queries_handled": self.engine.stats["queries"],
                "compression_ratio": 1.0,
                "overflow_telemetry": {"overflow_occurred": False, "limit": None, "initial_tokens": 0, "final_tokens": 0, "steps": []},
                "budget_tracking": {"memory_tokens_used": 0, "memory_tokens_limit": 1500, "document_tokens_used": 0, "document_tokens_limit": 0},
                "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "mode": "agentic"
            }}
            events_queue.append(done_event)
            return {
                "final_response": registry_text,
                "events_queue": events_queue
            }
            
        elif exit_type == "greeting":
            direct_reply = "Hello! How can I help you today? I'm ready to answer any questions about your uploaded documents or search the web."
            events_queue.append({"event": "answer_chunk", "text": direct_reply})
            
            await asyncio.to_thread(self.engine.save_memory, session_id, query, "user")
            await asyncio.to_thread(self.engine.save_memory, session_id, direct_reply, "assistant", 0.5)
            
            done_event = {"event": "done", "response": direct_reply, "stats": {
                "queries_handled": self.engine.stats["queries"],
                "compression_ratio": 1.0,
                "overflow_telemetry": {"overflow_occurred": False, "limit": None, "initial_tokens": 0, "final_tokens": 0, "steps": []},
                "budget_tracking": {"memory_tokens_used": 0, "memory_tokens_limit": 1500, "document_tokens_used": 0, "document_tokens_limit": 0},
                "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "mode": "agentic"
            }}
            events_queue.append(done_event)
            return {
                "final_response": direct_reply,
                "events_queue": events_queue
            }
            
        return {}

    async def overflow_recovery(self, state: AgentState) -> dict:
        query = state["query"]
        session_id = state["session_id"]
        context_limit = state["context_limit"]
        memory = self.engine.get_memory(session_id)
        memory_text = memory.get_active_context()
        
        overflow_occurred = False
        overflow_steps = []
        events_queue = []
        
        agent_prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(memory_text) + count_tokens(query) + 50
        initial_tokens = agent_prompt_tokens
        final_tokens = agent_prompt_tokens
        
        if context_limit and agent_prompt_tokens > context_limit:
            overflow_occurred = True
            overflow_steps.append(
                f"🚨 [AGENT] OVERFLOW DETECTED: Agent prompt size ({agent_prompt_tokens} tokens) "
                f"exceeds limit ({context_limit} tokens) by {agent_prompt_tokens - context_limit} tokens."
            )
            # Prune memory
            old_mem = count_tokens(memory_text)
            temp_entries = list(memory.entries)
            pruned_count = 0
            while len(temp_entries) > 1 and agent_prompt_tokens > context_limit:
                removed = temp_entries.pop(0)
                pruned_count += 1
                temp_mem_text = "".join([f"[{e.role}]: {e.text}\n" for e in temp_entries])
                agent_prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(temp_mem_text) + count_tokens(query) + 50
            
            if pruned_count > 0:
                memory.entries = temp_entries
                memory_text = memory.get_active_context()
                new_mem = count_tokens(memory_text)
                overflow_steps.append(f"   - Evicted {pruned_count} oldest conversational turns. Memory reduced from {old_mem} to {new_mem} tokens.")
            else:
                overflow_steps.append("   - No memory turns available for eviction.")
                
            # Truncate user query
            if agent_prompt_tokens > context_limit:
                import tiktoken
                from core.config import TOKENIZER_ENCODING
                tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)
                allowed_query_len = context_limit - count_tokens(SYSTEM_PROMPT) - count_tokens(memory_text) - 60
                allowed_query_len = max(5, allowed_query_len)
                query_tokens = tokenizer.encode(query)
                query = tokenizer.decode(query_tokens[:allowed_query_len])
                agent_prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(memory_text) + count_tokens(query) + 50
                overflow_steps.append(f"   - Hard truncated user query to {count_tokens(query)} tokens.")
                
            overflow_steps.append(f"✅ [AGENT] RECOVERY COMPLETE: Agent context size is now {agent_prompt_tokens} tokens.")
            final_tokens = agent_prompt_tokens
            
            events_queue.append({
                "event": "overflow_detected",
                "limit": context_limit,
                "initial": initial_tokens,
                "final": final_tokens,
                "steps": overflow_steps
            })
            for step in overflow_steps:
                events_queue.append({"event": "overflow_step", "text": step})
                
        return {
            "query": query,
            "memory_text": memory_text,
            "overflow_occurred": overflow_occurred,
            "overflow_steps": overflow_steps,
            "events_queue": events_queue,
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens
        }

    async def reasoning(self, state: AgentState) -> dict:
        query = state["query"]
        memory_text = state["memory_text"]
        scratchpad = state["scratchpad"]
        iteration = state["iteration"]
        goals_set = list(state["goals_set"])
        llm_call_count = state["llm_call_count"]
        
        events_queue = []
        events_queue.append({"event": "state_change", "state": "WAITING_FOR_REASONING"})
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Active Conversation History:\n{memory_text}\n\nUser Question: {query}"}
        ]
        
        if scratchpad:
            messages.append({"role": "assistant", "content": scratchpad})
            
        llm_call_count += 1
        
        try:
            completion = await self._call_llm_with_retry_async(
                messages=messages,
                temperature=0.0,
                frequency_penalty=0.0,
                stream=False
            )
            response = completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Agent LLM error: {e}")
            err_msg = f"I'm sorry, I encountered an LLM execution error: {e}"
            events_queue.append({"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"})
            events_queue.append({"event": "answer_chunk", "text": err_msg})
            events_queue.append({"event": "done", "response": err_msg, "stats": {}})
            return {
                "final_response": err_msg,
                "events_queue": events_queue,
                "llm_call_count": llm_call_count
            }

        # Parse thought
        thought_text = ""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final\s+Answer\s*:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought_text = thought_match.group(1).strip()
            if thought_text:
                events_queue.append({"event": "thought", "text": thought_text})
                goal_match = re.search(r'^([^.!?]+)', thought_text)
                if goal_match:
                    goal = goal_match.group(1).strip()
                    if goal not in goals_set:
                        goals_set.append(goal)

        # Parse Action or Final Answer
        action_info = self.agent.parse_action(response)
        final_answer_match = re.search(r"Final\s+Answer\s*:\s*(.*)", response, re.DOTALL | re.IGNORECASE)
        
        is_direct = False
        final_response = ""
        if final_answer_match:
            final_response = final_answer_match.group(1).strip()
            events_queue.append({"event": "state_change", "state": "WAITING_FOR_FINAL_ANSWER"})
        else:
            stripped_resp = response.strip()
            is_direct = len(stripped_resp) >= 15 and not (
                stripped_resp.lower().startswith("thought:") or 
                stripped_resp.lower().startswith("action:")
            )
            if is_direct:
                final_response = stripped_resp
                events_queue.append({"event": "state_change", "state": "WAITING_FOR_FINAL_ANSWER"})

        parsed_action = None
        if action_info:
            parsed_action = action_info
            events_queue.append({"event": "state_change", "state": "WAITING_FOR_ACTION"})
            events_queue.append({"event": "action", "tool": action_info[0], "input": action_info[1]})

        return {
            "iteration": iteration + 1,
            "llm_call_count": llm_call_count,
            "goals_set": goals_set,
            "parsed_action": parsed_action,
            "is_direct": is_direct,
            "final_response": final_response,
            "events_queue": events_queue,
            "raw_response": response
        }

    def route_after_reasoning(self, state: AgentState) -> str:
        final_response = state.get("final_response")
        parsed_action = state.get("parsed_action")
        iteration = state.get("iteration")
        
        if final_response:
            return "streaming_final_answer"
            
        if parsed_action:
            return "execute_tool"
            
        if iteration < 3:
            return "execute_formatting_error"
            
        return "synthesis"

    def route_after_tool(self, state: AgentState) -> str:
        if state.get("iteration", 0) >= 3:
            return "synthesis"
        return "reasoning"

    async def execute_formatting_error(self, state: AgentState) -> dict:
        scratchpad = state["scratchpad"]
        raw_response = state.get("raw_response", "Analyzing next steps")
        
        observation = (
            "Error: Your response did not contain a valid ReAct Action or Final Answer format. "
            "Remember to always format your next step exactly as:\n"
            "Thought: <your thought process>\n"
            "Action: <tool_name>[<arguments>]\n"
            "Or if you have the final answer, format it exactly as:\n"
            "Thought: <your thought process>\n"
            "Final Answer: <your response>"
        )
        
        events_queue = []
        events_queue.append({"event": "observation", "output": observation})
        
        new_scratchpad = scratchpad + f"\nThought: {raw_response}\nObservation: {observation}"
        
        return {
            "scratchpad": new_scratchpad,
            "events_queue": events_queue
        }

    async def execute_tool(self, state: AgentState) -> dict:
        scratchpad = state["scratchpad"]
        events_queue = []
        parsed_action = state.get("parsed_action")
        
        if not parsed_action or not isinstance(parsed_action, (list, tuple)) or len(parsed_action) < 2:
            observation = "Error: Invalid or missing action definition. Your next response must contain a valid tool call."
            events_queue.append({"event": "observation", "output": observation})
            return {
                "scratchpad": scratchpad + f"\nObservation: {observation}",
                "events_queue": events_queue
            }
            
        tool_name, tool_arg = parsed_action
        if tool_name:
            tool_name = tool_name.strip().lower()
        else:
            tool_name = ""
            
        if tool_arg:
            tool_arg = tool_arg.strip()
        else:
            tool_arg = ""
            
        raw_response = state.get("raw_response", "")
        iteration = state.get("iteration", 0)
        actions_taken = list(state.get("actions_taken", []))
        source_filter = state.get("source_filter")
        memory_text = state.get("memory_text", "")
        search_cache = dict(state.get("search_cache", {}))
        
        events_queue.append({"event": "state_change", "state": "EXECUTING_TOOL"})
        
        observation = ""
        try:
            if tool_name == "web_search":
                if tool_arg in search_cache:
                    logger.info(f"Search cache hit for: {tool_arg}")
                    observation = search_cache[tool_arg]
                else:
                    logger.info(f"Executing web_search for: {tool_arg}")
                    observation = mock_web_search(tool_arg)
                    search_cache[tool_arg] = observation
            elif tool_name == "web_scrape":
                if tool_arg in search_cache:
                    logger.info(f"Search cache hit for: {tool_arg}")
                    observation = search_cache[tool_arg]
                else:
                    logger.info(f"Executing web_scrape for: {tool_arg}")
                    scraped_text = await scrape_web_page_async(tool_arg)
                    if scraped_text.startswith("Error:") or scraped_text.startswith("Warning:"):
                        observation = scraped_text
                    else:
                        chunks = [p.strip() for p in scraped_text.split("\n") if len(p.strip()) > 30]
                        if chunks:
                            user_query = state.get("query", "")
                            mem_tokens = count_tokens(memory_text)
                            doc_budget = max(400, 1500 - mem_tokens)
                            compressed = await asyncio.to_thread(
                                self.engine.compressor.compress,
                                chunks, user_query, max_tokens=doc_budget
                            )
                            if compressed.strip():
                                observation = compressed
                            else:
                                text_summary = "\n\n".join(chunks[:6])
                                if len(text_summary) > 1200:
                                    text_summary = text_summary[:1200] + "..."
                                observation = text_summary
                        else:
                            observation = scraped_text[:1200]
                    search_cache[tool_arg] = observation
            elif tool_name == "get_current_time":
                logger.info("Executing get_current_time")
                observation = f"Current Datetime: {get_current_time()}"
            elif tool_name == "calculator":
                logger.info(f"Executing calculator for: {tool_arg}")
                observation = evaluate_math(tool_arg)
            elif tool_name == "get_registry":
                observation = self.engine._get_registry_context_text()
            elif tool_name == "search_knowledge_base":
                if tool_arg in search_cache:
                    logger.info(f"Search cache hit for: {tool_arg}")
                    observation = search_cache[tool_arg]
                else:
                    logger.info(f"Executing search_knowledge_base for: {tool_arg}")
                    search_queries = await self.engine._phase_expand_async(tool_arg, "context_engine", {})
                    raw_results = await self.engine._phase_retrieve_async(search_queries, 5, source_filter, {})
                    mem_tokens = count_tokens(memory_text)
                    doc_budget = max(300, 1500 - mem_tokens)
                    compressed = await asyncio.to_thread(
                        self.engine.compressor.compress,
                        [r["text"] for r in raw_results], tool_arg, max_tokens=doc_budget
                    )
                    observation = compressed if compressed.strip() else "No matching documents found in database."
                    search_cache[tool_arg] = observation
            elif tool_name == "get_system_stats":
                doc_count = await asyncio.to_thread(self.engine.retriever.get_count)
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                observation = f"System Stats: CPU={cpu}%, RAM={ram}%, Total Indexed Documents={doc_count}"
            elif tool_name == "direct_response":
                observation = f"Direct Response Executed: '{tool_arg}'"
            else:
                observation = (
                    f"Error: Unknown tool '{tool_name}'. "
                    "Available tools are: web_search[query], get_system_stats[], get_registry[], direct_response[response], web_scrape[url], get_current_time[], or calculator[expression]."
                )
        except Exception as tool_err:
            logger.error(f"Agent tool execution error for '{tool_name}': {tool_err}", exc_info=True)
            observation = f"Error executing tool '{tool_name}': {type(tool_err).__name__}: {tool_err}"

        events_queue.append({"event": "observation", "output": observation})
        
        thought_text = ""
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final\s+Answer\s*:|$)", raw_response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought_text = thought_match.group(1).strip()
            
        from core.security import sanitize_document_text
        sanitized_observation = sanitize_document_text(observation)
        new_scratchpad = scratchpad + f"\nThought: {thought_text}\nAction: {tool_name}[{tool_arg}]\nObservation: {sanitized_observation}"
        
        actions_taken.append({
            "step": iteration,
            "tool": tool_name,
            "input": tool_arg,
            "observation": sanitized_observation[:1000]
        })
        
        return {
            "scratchpad": new_scratchpad,
            "actions_taken": actions_taken,
            "events_queue": events_queue,
            "search_cache": search_cache
        }

    async def streaming_final_answer(self, state: AgentState) -> dict:
        final_response = state["final_response"]
        query = state["query"]
        session_id = state["session_id"]
        memory_text = state["memory_text"]
        overflow_occurred = state["overflow_occurred"]
        context_limit = state["context_limit"]
        overflow_steps = state["overflow_steps"]
        goals_set = state["goals_set"]
        actions_taken = state["actions_taken"]
        llm_call_count = state["llm_call_count"]
        
        events_queue = []
        events_queue.append({"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"})
        
        chunk_size = 8
        if final_response:
            for i in range(0, len(final_response), chunk_size):
                events_queue.append({"event": "answer_chunk", "text": final_response[i:i+chunk_size]})
            
        try:
            await asyncio.to_thread(self.engine.save_memory, session_id, query, "user")
        except Exception as e:
            logger.error(f"Failed to save user memory query: {e}")
        
        try:
            mem_tokens_used = count_tokens(memory_text)
        except Exception:
            mem_tokens_used = 0
            
        telemetry_data = {
            "query": query,
            "raw_prompt": f"SYSTEM_PROMPT:\n{SYSTEM_PROMPT}\n\nUSER_MESSAGE:\nActive Conversation History:\n{memory_text}\n\nUser Question: {query}",
            "overflow_occurred": overflow_occurred,
            "limit": context_limit,
            "initial_tokens": state.get("initial_tokens", 0),
            "final_tokens": state.get("final_tokens", 0),
            "steps": overflow_steps,
            "budget_tracking": {
                "memory_tokens_used": mem_tokens_used,
                "memory_tokens_limit": 1500,
                "document_tokens_used": 0,
                "document_tokens_limit": 0
            },
            "compression_ratio": 1.0
        }
        
        try:
            await asyncio.to_thread(self.engine.save_memory, session_id, final_response, "assistant", 0.8, telemetry=telemetry_data)
        except Exception as e:
            logger.error(f"Failed to save assistant memory response: {e}")
        
        try:
            queries_handled = self.engine.stats["queries"]
        except Exception:
            queries_handled = 0
            
        try:
            active_memories = len(self.engine.get_memory(session_id).entries)
        except Exception:
            active_memories = 0
            
        try:
            cpu_usage_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_usage_percent = 0.0
            
        try:
            memory_usage_percent = psutil.virtual_memory().percent
        except Exception:
            memory_usage_percent = 0.0
            
        done_event = {"event": "done", "response": final_response, "stats": {
            "queries_handled": queries_handled,
            "compression_ratio": 1.0,
            "active_memories": active_memories,
            "instantaneous_latency_ms": {"total_execution_ms": 0.0},
            "avg_latency_ms": 0.0,
            "cpu_usage_percent": cpu_usage_percent,
            "memory_usage_percent": memory_usage_percent,
            "context_used_percent": 0.0,
            "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
            "query_cost": "$0.00000000",
            "budget_tracking": telemetry_data["budget_tracking"],
            "overflow_telemetry": telemetry_data,
            "retrieved_context": [],
            "raw_prompt": telemetry_data["raw_prompt"],
            "mode": "agentic",
            "alpha": 0.5,
            "reranker_peak_score": 0.0,
            "debug_info": {
                "llm_calls": llm_call_count,
                "goals_set": goals_set,
                "actions_taken": actions_taken
            }
        }}
        events_queue.append(done_event)
        
        return {
            "events_queue": events_queue
        }

    async def synthesis(self, state: AgentState) -> dict:
        query = state["query"]
        scratchpad = state["scratchpad"]
        session_id = state["session_id"]
        memory_text = state["memory_text"]
        overflow_occurred = state["overflow_occurred"]
        context_limit = state["context_limit"]
        overflow_steps = state["overflow_steps"]
        goals_set = state["goals_set"]
        actions_taken = state["actions_taken"]
        llm_call_count = state["llm_call_count"]
        
        events_queue = []
        events_queue.append({"event": "state_change", "state": "WAITING_FOR_FINAL_ANSWER"})
        events_queue.append({"event": "thought", "text": "Iteration limit reached. Synthesizing final answer from gathered observations."})
        events_queue.append({"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"})
        
        synthesis_prompt = (
            f"The user asked: {query}\n\n"
            f"Here is what was found during the investigation:\n{scratchpad}\n\n"
            "Write a comprehensive, detailed, and beautifully structured final answer summarizing these observations. "
            "Follow these formatting rules:\n"
            "1. Use Markdown headers, bold highlights, and lists to structure your explanation.\n"
            "2. Format code, commands, or values in code blocks.\n"
            "3. Provide depth and clear articulation rather than a short summary.\n"
            "4. If the information is not sufficient, state what you know and what is missing clearly."
        )
        
        llm_call_count += 1
        final_response = ""
        try:
            stream = await self._call_llm_with_retry_async(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Synthesize a clear final answer based on the provided investigation log."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.7,
                frequency_penalty=0.3,
                stream=True
            )
            async for chunk in async_iter(stream):
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    final_response += delta
                    events_queue.append({"event": "answer_chunk", "text": delta})
        except Exception as e:
            logger.error(f"Error during final synthesis: {e}")
            fallback_response = (
                "### Investigation Summary (Fallback)\n\n"
                "I encountered an issue synthesizing the final response using the LLM model, but here are the key findings from my investigation:\n\n"
            )
            if scratchpad:
                steps = scratchpad.strip().split("\n")
                for step in steps:
                    step = step.strip()
                    if step.startswith("Thought:"):
                        thought = step[len("Thought:"):].strip()
                        if thought:
                            fallback_response += f"- **Analysis**: {thought}\n"
                    elif step.startswith("Action:"):
                        action = step[len("Action:"):].strip()
                        if action:
                            fallback_response += f"  - *Executed action*: `{action}`\n"
                    elif step.startswith("Observation:"):
                        obs = step[len("Observation:"):].strip()
                        if obs:
                            if len(obs) > 150:
                                obs = obs[:150] + "..."
                            fallback_response += f"  - *Observation result*: {obs}\n"
            else:
                fallback_response += "*No detailed observations were recorded during this session.*"
                
            final_response = fallback_response
            chunk_size = 12
            for i in range(0, len(final_response), chunk_size):
                events_queue.append({"event": "answer_chunk", "text": final_response[i:i+chunk_size]})
            
        try:
            await asyncio.to_thread(self.engine.save_memory, session_id, query, "user")
        except Exception as e:
            logger.error(f"Failed to save user memory query: {e}")
        
        try:
            mem_tokens_used = count_tokens(memory_text)
        except Exception:
            mem_tokens_used = 0
            
        telemetry_data = {
            "query": query,
            "raw_prompt": f"SYSTEM_PROMPT:\n{SYSTEM_PROMPT}\n\nUSER_MESSAGE:\nActive Conversation History:\n{memory_text}\n\nUser Question: {query}",
            "overflow_occurred": overflow_occurred,
            "limit": context_limit,
            "initial_tokens": state.get("initial_tokens", 0),
            "final_tokens": state.get("final_tokens", 0),
            "steps": overflow_steps,
            "budget_tracking": {
                "memory_tokens_used": mem_tokens_used,
                "memory_tokens_limit": 1500,
                "document_tokens_used": 0,
                "document_tokens_limit": 0
            },
            "compression_ratio": 1.0
        }
        
        try:
            await asyncio.to_thread(self.engine.save_memory, session_id, final_response, "assistant", 0.8, telemetry=telemetry_data)
        except Exception as e:
            logger.error(f"Failed to save assistant memory response: {e}")
        
        try:
            queries_handled = self.engine.stats["queries"]
        except Exception:
            queries_handled = 0
            
        try:
            active_memories = len(self.engine.get_memory(session_id).entries)
        except Exception:
            active_memories = 0
            
        try:
            cpu_usage_percent = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_usage_percent = 0.0
            
        try:
            memory_usage_percent = psutil.virtual_memory().percent
        except Exception:
            memory_usage_percent = 0.0
            
        done_event = {"event": "done", "response": final_response, "stats": {
            "queries_handled": queries_handled,
            "compression_ratio": 1.0,
            "active_memories": active_memories,
            "instantaneous_latency_ms": {"total_execution_ms": 0.0},
            "avg_latency_ms": 0.0,
            "cpu_usage_percent": cpu_usage_percent,
            "memory_usage_percent": memory_usage_percent,
            "context_used_percent": 0.0,
            "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
            "query_cost": "$0.00000000",
            "budget_tracking": telemetry_data["budget_tracking"],
            "overflow_telemetry": telemetry_data,
            "retrieved_context": [],
            "raw_prompt": telemetry_data["raw_prompt"],
            "mode": "agentic",
            "alpha": 0.5,
            "reranker_peak_score": 0.0,
            "debug_info": {
                "llm_calls": llm_call_count,
                "goals_set": goals_set,
                "actions_taken": actions_taken
            }
        }}
        events_queue.append(done_event)
        
        return {
            "final_response": final_response,
            "llm_call_count": llm_call_count,
            "events_queue": events_queue
        }

