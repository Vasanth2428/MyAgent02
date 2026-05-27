import re
import logging
import psutil
from typing import Dict, Generator, Optional, List
import tiktoken
from core.config import TOKENIZER_ENCODING

logger = logging.getLogger("RAG.Agent")
tokenizer = tiktoken.get_encoding(TOKENIZER_ENCODING)

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text))


def mock_web_search(query: str) -> str:
    """Simulates a deterministic, offline web search response for web agent queries."""
    query_lower = query.lower()
    if "google" in query_lower:
        return (
            "Web Search Results for 'Google':\n"
            "- Google Cloud announces new AI infrastructure accelerators (May 2026).\n"
            "- Alphabet Inc. stock rises 2.4% following strong quarterly earnings report.\n"
            "- Google DeepMind details new agentic code generation systems."
        )
    elif "nvidia" in query_lower:
        return (
            "Web Search Results for 'Nvidia':\n"
            "- Nvidia releases next-gen Blackwell Ultra GPU architectures for data centers.\n"
            "- NVDA stock reaches new highs as demand for generative AI training chips persists.\n"
            "- Nvidia CEO speaks on the future of physical AI and robotics at GTC 2026."
        )
    elif "apple" in query_lower or "iphone" in query_lower:
        return (
            "Web Search Results for 'Apple':\n"
            "- Apple announces iOS 20 with deep neural agentic capabilities integrated at the OS level.\n"
            "- Stock analysts upgrade AAPL following strong global iPhone shipments.\n"
            "- Apple Vision Pro 2 rumored to launch late next year with lighter design."
        )
    elif "weather" in query_lower:
        return (
            "Web Search Results for 'Weather':\n"
            "- National Weather Service issues high temperature advisories for major metropolitan areas.\n"
            "- Current global meteorological models predict an active tropical storm season.\n"
            "- Local weather: Mild temperatures, clear skies, and light winds expected over the weekend."
        )
    else:
        return (
            f"Web Search Results for '{query}':\n"
            f"- No official news articles found.\n"
            f"- Online forums indicate interest in: {query}.\n"
            f"- Search query returned index documents for general tech trends."
        )


SYSTEM_PROMPT = """You are an advanced, highly-intelligent Web & System Assistant.
You have access to system stats and web search tools to help provide authoritative, precise, and well-articulated answers to user queries.

You must solve the user's request step-by-step using a ReAct loop.
You must use the following format:

Thought: Write what you need to do next to answer the user query. Be critical, outline a plan, identify constraints, and think deeply.
Action: tool_name[arguments]
Observation: The output result from the tool.
... (this loop can repeat at most 3 times)
Thought: I have gathered enough information to write a comprehensive, final response.
Final Answer: Write a detailed, structured, and beautifully formatted response to the user.

Available tools:
1. web_search[query]: Searches the web for latest articles, news, or query information. Use this ONLY when the user asks for real-time news, latest web search information, or current affairs.
2. get_system_stats[]: Returns current CPU usage, RAM usage, and total indexed documents.
3. get_registry[]: Returns a listing of all indexed documents, files, datasets, databases, schemas, and sources. Use this when users ask "what documents do you contain", "list files", "show me sources", or similar queries about available knowledge.
4. direct_response[response]: Use this to respond directly to the user for general greetings, chit-chat, or if you can answer using the conversation history alone.

Strict format rules:
1. ONLY call one tool at a time.
2. You MUST use the exact format "Action: tool_name[arguments]". For example: "Action: web_search[latest news on Nvidia]" or "Action: get_system_stats[]".
3. Do NOT put quotes or backticks around tool arguments.
4. If the tools do not return enough relevant information, state that you do not know in the Final Answer.

Critical Response Quality & Formatting Guidelines:
1. Depth & Articulation: Never provide short, single-sentence answers when explaining complex topics. Always expand, provide context, explain "why" things are the way they are, and summarize your findings thoroughly.
2. Structure & Markdown: Organize your Final Answer beautifully. Use appropriate headers (### or ####), bullet points, and numbered lists to structure your response.
3. Highlighting: Use bold text to highlight key configuration values, commands, server names, and crucial definitions.
4. Technical Assets: Format code snippets, database commands, and logs inside proper syntax-highlighted markdown code blocks (e.g. ```sql, ```bash, ```python).
5. Tabular Data: If you need to present comparison lists or structured facts (like system stats), compile them using clean Markdown tables.
"""


class RAGAgent:
    """
    An autonomous ReAct agent running on top of RAGContextEngine.
    """

    def __init__(self, engine):
        self.engine = engine
        self.max_iterations = 3
        self._debug_mode = False

    def enable_debug(self, enabled: bool = True):
        """Enable/disable debug tracking for LLM calls and goal progression."""
        self._debug_mode = enabled

    def parse_action(self, text: str) -> Optional[tuple]:
        """Parses action from LLM response. E.g. Action: web_search[my query]"""
        match = re.search(r"Action:\s*([\w\-]+)\s*(?:\[(.*?)\])?", text, re.IGNORECASE | re.DOTALL)
        if match:
            tool_name = match.group(1).strip().lower()
            tool_arg = match.group(2) or ""
            tool_arg = tool_arg.strip()
            # Strip outer quotes/backticks if present
            if len(tool_arg) >= 2 and (
                (tool_arg.startswith('"') and tool_arg.endswith('"')) or
                (tool_arg.startswith("'") and tool_arg.endswith("'")) or
                (tool_arg.startswith("`") and tool_arg.endswith("`"))
            ):
                tool_arg = tool_arg[1:-1].strip()
            return tool_name, tool_arg
        return None

    def run_stream(self, query: str, session_id: str = "default", source_filter: Optional[str] = None, context_limit: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Executes the ReAct loop deterministically and yields events for streaming.
        Uses non-streaming reasoning (temp=0.0) and streams only final response.
        """
        import time
        logger.info(f"Agent starting for query: {query[:50]}... | Limit: {context_limit}")
        memory = self.engine.get_memory(session_id)
        memory_text = memory.get_active_context()

        # Early exit check: greetings or registry queries
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
        
        if is_registry_query:
            logger.info("Agent early exit: Registry query detected.")
            yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
            yield {"event": "thought", "text": "This query asks about available documents. I'll retrieve the registry listing directly."}
            registry_text = self.engine._get_registry_context_text()
            for i in range(0, len(registry_text), 20):
                yield {"event": "answer_chunk", "text": registry_text[i:i+20]}
                time.sleep(0.01)
            self.engine.save_memory(session_id, query, "user")
            self.engine.save_memory(session_id, registry_text, "assistant", 0.8)
            yield {"event": "done", "response": registry_text, "stats": {
                "queries_handled": self.engine.stats["queries"],
                "compression_ratio": 1.0,
                "overflow_telemetry": {"overflow_occurred": False, "limit": None, "initial_tokens": 0, "final_tokens": 0, "steps": []},
                "budget_tracking": {"memory_tokens_used": 0, "memory_tokens_limit": 1500, "document_tokens_used": 0, "document_tokens_limit": 0},
                "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "mode": "agentic"
            }}
            return
        
        if clean_query in greetings or len(clean_query.split()) <= 2 and not any(w in clean_query for w in ["ip", "password", "redis", "database", "stats", "limit", "config", "version", "target"]):
            logger.info("Agent early exit: Simple greeting/chat.")
            yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
            yield {"event": "thought", "text": "This is a simple query or greeting. I can respond directly without searching the web."}
            direct_reply = f"Hello! How can I help you today? I'm ready to answer any questions about your uploaded documents or search the web."
            yield {"event": "answer_chunk", "text": direct_reply}
            self.engine.save_memory(session_id, query, "user")
            self.engine.save_memory(session_id, direct_reply, "assistant", 0.5)
            yield {"event": "done", "response": direct_reply, "stats": {
                "queries_handled": self.engine.stats["queries"],
                "compression_ratio": 1.0,
                "overflow_telemetry": {"overflow_occurred": False, "limit": None, "initial_tokens": 0, "final_tokens": 0, "steps": []},
                "budget_tracking": {"memory_tokens_used": 0, "memory_tokens_limit": 1500, "document_tokens_used": 0, "document_tokens_limit": 0},
                "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
                "mode": "agentic"
            }}
            return

        # Perform overflow detection
        overflow_occurred = False
        overflow_steps = []
        agent_prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(memory_text) + count_tokens(query) + 50
        initial_tokens = agent_prompt_tokens

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
                allowed_query_len = context_limit - count_tokens(SYSTEM_PROMPT) - count_tokens(memory_text) - 60
                allowed_query_len = max(5, allowed_query_len)
                query_tokens = tokenizer.encode(query)
                query = tokenizer.decode(query_tokens[:allowed_query_len])
                agent_prompt_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(memory_text) + count_tokens(query) + 50
                overflow_steps.append(f"   - Hard truncated user query to {count_tokens(query)} tokens.")
                
            overflow_steps.append(f"✅ [AGENT] RECOVERY COMPLETE: Agent context size is now {agent_prompt_tokens} tokens.")
            
            yield {
                "event": "overflow_detected",
                "limit": context_limit,
                "initial": initial_tokens,
                "final": agent_prompt_tokens,
                "steps": overflow_steps
            }
            for step in overflow_steps:
                yield {"event": "overflow_step", "text": step}
                time.sleep(0.4)

        # Initialize scratchpad
        scratchpad = ""
        search_cache = {}

        # Debug tracking (isolated from response stream)
        llm_call_count = 0
        goals_set = []  # Goals extracted from thoughts
        actions_taken = []  # Actions and their outcomes

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Active Conversation History:\n{memory_text}\n\nUser Question: {query}"}
        ]

        final_response = ""
        last_thought = ""
        any_answer_streamed = False

        for iteration in range(self.max_iterations):
            llm_call_count += 1  # Count each LLM call (reasoning step)
            yield {"event": "state_change", "state": "WAITING_FOR_REASONING"}
            
            # Update messages with scratchpad history
            current_messages = list(messages)
            if scratchpad:
                current_messages.append({"role": "assistant", "content": scratchpad})

            # Call LLM synchronously (stream=False, temperature=0.0)
            logger.info(f"Agent Iteration {iteration+1} calling LLM (sync reasoning)...")
            
            try:
                completion = self.engine.client.chat.completions.create(
                    model=self.engine.llm_service.model,
                    messages=current_messages,
                    temperature=0.0, # Deterministic reasoning
                    frequency_penalty=0.0,
                    stream=False
                )
                response = completion.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Agent LLM error: {e}")
                err_msg = f"I'm sorry, I encountered an LLM execution error: {e}"
                yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
                yield {"event": "answer_chunk", "text": err_msg}
                yield {"event": "done", "response": err_msg, "stats": {}}
                return

            logger.info(f"Agent response:\n{response}")

            # Parse Thought
            thought_text = ""
            thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final\s+Answer\s*:|$)", response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought_text = thought_match.group(1).strip()
                if thought_text and thought_text != last_thought:
                    yield {"event": "thought", "text": thought_text}
                    last_thought = thought_text
                    # Extract goal: first sentence of thought (typically states the objective)
                    goal_match = re.search(r'^([^.!?]+)', thought_text)
                    if goal_match:
                        goal = goal_match.group(1).strip()
                        if goal not in goals_set:
                            goals_set.append(goal)

            # Parse Action or Final Answer
            action_info = self.parse_action(response)
            final_answer_match = re.search(r"Final\s+Answer\s*:\s*(.*)", response, re.DOTALL | re.IGNORECASE)

            if final_answer_match:
                final_response = final_answer_match.group(1).strip()
                yield {"event": "state_change", "state": "WAITING_FOR_FINAL_ANSWER"}
                break
            
            # If the response looks like a direct reply without tools or Final Answer tags
            # (e.g. it does not start with Thought: or Action:)
            stripped_resp = response.strip()
            is_direct = len(stripped_resp) >= 15 and not (
                stripped_resp.lower().startswith("thought:") or 
                stripped_resp.lower().startswith("action:")
            )
            
            if is_direct:
                final_response = stripped_resp
                yield {"event": "state_change", "state": "WAITING_FOR_FINAL_ANSWER"}
                break

            if action_info:
                tool_name, tool_arg = action_info
                yield {"event": "state_change", "state": "WAITING_FOR_ACTION"}
                yield {"event": "action", "tool": tool_name, "input": tool_arg}

                # Execute Tool
                yield {"event": "state_change", "state": "EXECUTING_TOOL"}
                observation = ""
                try:
                    if tool_name == "web_search":
                        # Check cache first
                        if tool_arg in search_cache:
                            logger.info(f"Search cache hit for: {tool_arg}")
                            observation = search_cache[tool_arg]
                        else:
                            logger.info(f"Executing web_search for: {tool_arg}")
                            observation = mock_web_search(tool_arg)
                            search_cache[tool_arg] = observation
                    elif tool_name == "get_registry":
                        logger.info("Executing get_registry tool")
                        try:
                            summary = self.engine.registry.get_registry_summary()
                            observation = f"Registry Summary: {len(summary.get('sources', []))} sources indexed."
                            if summary.get('sources'):
                                observation += f" Sources: {', '.join(summary['sources'][:10])}"
                            registry_text = self.engine._get_registry_context_text()
                            observation = registry_text
                        except Exception as reg_err:
                            observation = f"Registry lookup error: {reg_err}"

                    elif tool_name == "search_knowledge_base":
                        # Legacy compatibility for search_knowledge_base unit tests
                        if tool_arg in search_cache:
                            logger.info(f"Search cache hit for: {tool_arg}")
                            observation = search_cache[tool_arg]
                        else:
                            logger.info(f"Executing search_knowledge_base for: {tool_arg}")
                            search_queries = self.engine._phase_expand(tool_arg, "context_engine", {})
                            raw_results = self.engine._phase_retrieve(search_queries, 5, source_filter, {})
                            mem_tokens = count_tokens(memory_text)
                            doc_budget = max(300, 1500 - mem_tokens)
                            compressed = self.engine.compressor.compress(
                                [r["text"] for r in raw_results], tool_arg, max_tokens=doc_budget
                            )
                            observation = compressed if compressed.strip() else "No matching documents found in database."
                            search_cache[tool_arg] = observation

                    elif tool_name == "get_system_stats":
                        logger.info("Executing get_system_stats")
                        doc_count = self.engine.retriever.get_count()
                        cpu = psutil.cpu_percent(interval=None)
                        ram = psutil.virtual_memory().percent
                        observation = f"System Stats: CPU={cpu}%, RAM={ram}%, Total Indexed Documents={doc_count}"

                    elif tool_name == "direct_response":
                        logger.info("Executing direct_response tool")
                        observation = f"Direct Response Executed: '{tool_arg}'"
                        final_response = tool_arg
                        yield {"event": "thought", "text": f"Direct response decided: {tool_arg}"}
                        break
                    else:
                        observation = (
                            f"Error: Unknown tool '{tool_name}'. "
                            "Available tools are: web_search[query], get_system_stats[], get_registry[], or direct_response[response]."
                        )
                except Exception as tool_err:
                    logger.error(f"Agent tool execution error for '{tool_name}': {tool_err}", exc_info=True)
                    observation = f"Error executing tool '{tool_name}': {type(tool_err).__name__}: {tool_err}"

                yield {"event": "observation", "output": observation}
                scratchpad += f"\nThought: {thought_text}\nAction: {tool_name}[{tool_arg}]\nObservation: {observation}"
                actions_taken.append({
                    "step": iteration + 1,
                    "tool": tool_name, 
                    "input": tool_arg, 
                    "observation": observation[:1000]
                })

            else:
                # If we have iterations left, feed a formatting error observation to self-correct
                if iteration < self.max_iterations - 1:
                    observation = (
                        "Error: Your response did not contain a valid ReAct Action or Final Answer format. "
                        "Remember to always format your next step exactly as:\n"
                        "Thought: <your thought process>\n"
                        "Action: <tool_name>[<arguments>]\n"
                        "Or if you have the final answer, format it exactly as:\n"
                        "Thought: <your thought process>\n"
                        "Final Answer: <your response>"
                    )
                    yield {"event": "observation", "output": observation}
                    scratchpad += f"\nThought: {thought_text or 'Analyzing next steps'}\nObservation: {observation}"
                    continue
                else:
                    final_response = response.strip()
                    break

        # If loop limit exhausted without generating final response, run final synthesis
        if not final_response:
            logger.info("Agent iteration limit reached. Generating final synthesis answer...")
            yield {"event": "state_change", "state": "WAITING_FOR_FINAL_ANSWER"}
            yield {"event": "thought", "text": "Iteration limit reached. Synthesizing final answer from gathered observations."}
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
            
            yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
            llm_call_count += 1  # Count synthesis LLM call
            try:
                # Call synthesis in streaming conversational mode (temp=0.7)
                stream = self.engine.client.chat.completions.create(
                    model=self.engine.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Synthesize a clear final answer based on the provided investigation log."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    temperature=0.7, # Conversational synthesis
                    frequency_penalty=0.3,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        final_response += delta
                        yield {"event": "answer_chunk", "text": delta}
                        any_answer_streamed = True
            except Exception as e:
                logger.error(f"Error during final synthesis: {e}")
                final_response = "I encountered an issue synthesizing the final answer from observations."
                yield {"event": "answer_chunk", "text": final_response}
                any_answer_streamed = True
        else:
            # Stream the final response chunk-by-chunk using typewriter effect
            yield {"event": "state_change", "state": "STREAMING_FINAL_RESPONSE"}
            chunk_size = 8
            for i in range(0, len(final_response), chunk_size):
                yield {"event": "answer_chunk", "text": final_response[i:i+chunk_size]}
                any_answer_streamed = True
                time.sleep(0.02)

        # Persist memory
        self.engine.save_memory(session_id, query, "user")
        
        telemetry_data = {
            "query": query,
            "raw_prompt": f"SYSTEM_PROMPT:\n{SYSTEM_PROMPT}\n\nUSER_MESSAGE:\nActive Conversation History:\n{memory_text}\n\nUser Question: {query}",
            "overflow_occurred": overflow_occurred,
            "limit": context_limit,
            "initial_tokens": initial_tokens,
            "final_tokens": agent_prompt_tokens,
            "steps": overflow_steps,
            "budget_tracking": {
                "memory_tokens_used": count_tokens(memory_text),
                "memory_tokens_limit": 1500,
                "document_tokens_used": 0,
                "document_tokens_limit": 0
            },
            "compression_ratio": 1.0
        }
        self.engine.save_memory(session_id, final_response, "assistant", 0.8, telemetry=telemetry_data)

        yield {"event": "done", "response": final_response, "stats": {
            "queries_handled": self.engine.stats["queries"],
            "compression_ratio": 1.0,
            "active_memories": len(memory.entries),
            "instantaneous_latency_ms": {"total_execution_ms": 0.0},
            "avg_latency_ms": 0.0,
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "context_used_percent": 0.0,
            "exact_tokens": {"prompt": 0, "completion": 0, "total": 0},
            "query_cost": "$0.00000000",
            "budget_tracking": telemetry_data.get("budget_tracking", {
                "memory_tokens_used": count_tokens(memory_text),
                "memory_tokens_limit": 1500,
                "document_tokens_used": 0,
                "document_tokens_limit": 0
            }),
            "overflow_telemetry": telemetry_data.get("overflow_telemetry", {
                "overflow_occurred": overflow_occurred,
                "limit": context_limit,
                "initial_tokens": initial_tokens,
                "final_tokens": agent_prompt_tokens,
                "steps": overflow_steps
            }),
            "retrieved_context": [],
            "raw_prompt": telemetry_data.get("raw_prompt", ""),
            "mode": "agentic",
            "alpha": 0.5,
            "reranker_peak_score": 0.0,
            "debug_info": {
                "llm_calls": llm_call_count,
                "goals_set": goals_set,
                "actions_taken": actions_taken
            }
        }}
