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


SYSTEM_PROMPT = """You are an advanced RAG Assistant with access to tools to help answer user questions.
You must solve the user's request step-by-step using a ReAct loop.
You must use the following format:

Thought: Write what you need to do next to answer the user query.
Action: tool_name[arguments]
Observation: The output result from the tool.
... (this loop can repeat at most 3 times)
Thought: I have enough information to write the final response.
Final Answer: Write the response to the user.

Available tools:
1. search_knowledge_base[query]: Searches the document database and returns compressed relevant segments. Use this when the query asks about technical facts, documentation, or upload files.
2. get_system_stats[]: Returns current CPU usage, RAM usage, and total indexed documents.
3. direct_response[response]: Use this to respond directly to the user for general greetings, chit-chat, or if you can answer using the conversation history alone.

Strict rules:
1. ONLY call one tool at a time.
2. You MUST use the exact format "Action: tool_name[arguments]". For example: "Action: search_knowledge_base[database password]" or "Action: get_system_stats[]".
3. Do NOT put quotes or backticks around tool arguments.
4. If the tools do not return enough relevant information, state that you do not know in the Final Answer.
"""

class RAGAgent:
    """
    An autonomous ReAct agent running on top of RAGContextEngine.
    """

    def __init__(self, engine):
        self.engine = engine
        self.max_iterations = 3

    def parse_action(self, text: str) -> Optional[tuple]:
        """Parses action from LLM response. E.g. Action: search_knowledge_base[my query]"""
        # Match "Action: tool_name[arguments]" or "Action: tool_name" with optional brackets
        match = re.search(r"Action:\s*([\w\-]+)\s*(?:\[(.*?)\])?", text, re.IGNORECASE)
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
        Executes the ReAct loop and yields intermediate events for streaming.
        """
        import time
        logger.info(f"Agent starting for query: {query[:50]}... | Limit: {context_limit}")
        memory = self.engine.get_memory(session_id)
        memory_text = memory.get_active_context()

        # Early exit check: greetings or extremely simple chat
        clean_query = query.lower().strip()
        greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are you"}
        if clean_query in greetings or len(clean_query.split()) <= 2 and not any(w in clean_query for w in ["ip", "password", "redis", "database", "stats", "limit", "config", "version", "target"]):
            logger.info("Agent early exit: Simple greeting/chat.")
            yield {"event": "thought", "text": "This is a simple query or greeting. I can respond directly without searching the knowledge base."}
            direct_reply = f"Hello! How can I help you today? I'm ready to answer any questions about your uploaded documents or system configuration."
            yield {"event": "answer_chunk", "text": direct_reply}
            self.engine.save_memory(session_id, query, "user")
            self.engine.save_memory(session_id, direct_reply, "assistant", 0.5)
            yield {"event": "done", "response": direct_reply, "stats": {}}
            return

        # Perform overflow detection on the Agent prompt (system prompt + memory + user query)
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
                
            # If still overflowing, truncate user query
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
        search_cache = {} # Simple in-memory cache to prevent redundant search calls

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Active Conversation History:\n{memory_text}\n\nUser Question: {query}"}
        ]

        final_response = ""
        last_thought = ""
        final_answer_streamed = False
        is_direct_answer = False

        for iteration in range(self.max_iterations):
            # Update messages with the scratchpad history of thoughts/actions
            current_messages = list(messages)
            if scratchpad:
                current_messages.append({"role": "assistant", "content": scratchpad})

            # Call LLM (Streaming)
            logger.info(f"Agent Iteration {iteration+1} calling LLM (streaming)...")
            response = ""
            final_answer_streamed = False
            is_direct_answer = False
            buffer = ""
            
            try:
                stream = self.engine.client.chat.completions.create(
                    model=self.engine.llm_service.model,
                    messages=current_messages,
                    temperature=0.1,
                    stream=True
                )
                
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    
                    response += delta
                    buffer += delta
                    
                    if final_answer_streamed or is_direct_answer:
                        yield {"event": "answer_chunk", "text": delta}
                        continue
                    
                    if "Final Answer:" in buffer:
                        final_answer_streamed = True
                        parts = buffer.split("Final Answer:", 1)
                        
                        # Yield Thought if it's there
                        thought_match = re.search(r"Thought:\s*(.*?)(?=Final Answer:|$)", parts[0], re.DOTALL)
                        if thought_match:
                            thought_text = thought_match.group(1).strip()
                            if thought_text and thought_text != last_thought:
                                yield {"event": "thought", "text": thought_text}
                                last_thought = thought_text
                        
                        answer_start = parts[1].strip()
                        if answer_start:
                            yield {"event": "answer_chunk", "text": answer_start}
                        buffer = ""
                        continue
                    
                    # Direct response fallback (doesn't follow ReAct, start streaming immediately)
                    stripped_buf = buffer.strip()
                    if len(stripped_buf) >= 15 and not (stripped_buf.startswith("Thought:") or stripped_buf.startswith("Action:")):
                        is_direct_answer = True
                        yield {"event": "answer_chunk", "text": buffer}
                        buffer = ""
                        continue
            except Exception as e:
                logger.error(f"Agent LLM error: {e}")
                err_msg = f"I'm sorry, I encountered an LLM execution error: {e}"
                yield {"event": "answer_chunk", "text": err_msg}
                yield {"event": "done", "response": err_msg, "stats": {}}
                return

            logger.info(f"Agent response:\n{response}")

            # Parse Thought
            thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
            if thought_match:
                thought_text = thought_match.group(1).strip()
                if thought_text and thought_text != last_thought:
                    yield {"event": "thought", "text": thought_text}
                    last_thought = thought_text

            # Parse Action or Final Answer
            action_info = self.parse_action(response)
            final_answer_match = re.search(r"Final Answer:\s*(.*)", response, re.DOTALL)

            if action_info:
                tool_name, tool_arg = action_info
                yield {"event": "action", "tool": tool_name, "input": tool_arg}

                # Execute Tool
                observation = ""
                try:
                    if tool_name == "search_knowledge_base":
                        # Check cache first
                        if tool_arg in search_cache:
                            logger.info(f"Search cache hit for: {tool_arg}")
                            observation = search_cache[tool_arg]
                        else:
                            logger.info(f"Executing search_knowledge_base for: {tool_arg}")
                            # Re-use engine's retrieve/refine steps for efficiency
                            search_queries = self.engine._phase_expand(tool_arg, "context_engine", {})
                            raw_results = self.engine._phase_retrieve(search_queries, 5, source_filter, {})
                            
                            # Compress with remaining budget
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
                        # Force break and set as final response
                        final_response = tool_arg
                        yield {"event": "thought", "text": f"Direct response decided: {tool_arg}"}
                        break
                    else:
                        observation = (
                            f"Error: Unknown tool '{tool_name}'. "
                            "Available tools are: search_knowledge_base[query], get_system_stats[], or direct_response[response]."
                        )
                except Exception as tool_err:
                    logger.error(f"Agent tool execution error for '{tool_name}': {tool_err}", exc_info=True)
                    observation = f"Error executing tool '{tool_name}': {type(tool_err).__name__}: {tool_err}"

                yield {"event": "observation", "output": observation}
                scratchpad += f"\nThought: {thought_text}\nAction: {tool_name}[{tool_arg}]\nObservation: {observation}"

            elif final_answer_match:
                final_response = final_answer_match.group(1).strip()
                break
            else:
                # If we have iterations left, feed a formatting error observation to the LLM to allow it to self-correct
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
                    t_text = thought_text if 'thought_text' in locals() and thought_text else "Analyzing next steps."
                    scratchpad += f"\nThought: {t_text}\nObservation: {observation}"
                    continue
                else:
                    # Fallback if model exhausted iterations without proper structure
                    final_response = response.strip()
                    break

        # If loop limit exhausted without generating final response, run final synthesis
        if not final_response:
            logger.info("Agent iteration limit reached. Generating final synthesis answer...")
            yield {"event": "thought", "text": "Iteration limit reached. Synthesizing final answer from gathered observations."}
            synthesis_prompt = f"The user asked: {query}\n\nHere is what was found during the investigation:\n{scratchpad}\n\nWrite a final answer to the user summarizing these observations. If the information is not sufficient, state what you know and what is missing."
            try:
                stream = self.engine.client.chat.completions.create(
                    model=self.engine.llm_service.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Synthesize a clear final answer based on the provided investigation log."},
                        {"role": "user", "content": synthesis_prompt}
                    ],
                    temperature=0.3,
                    stream=True
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        final_response += delta
                        yield {"event": "answer_chunk", "text": delta}
            except Exception as e:
                logger.error(f"Error during final synthesis: {e}")
                final_response = "I encountered an issue synthesizing the final answer from observations."
                yield {"event": "answer_chunk", "text": final_response}
        elif not final_answer_streamed and not is_direct_answer:
            # Yield final answer chunk-by-chunk to simulate streaming if not already streamed
            yield {"event": "answer_chunk", "text": final_response}

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
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "raw_prompt": f"SYSTEM_PROMPT:\n{SYSTEM_PROMPT}\n\nUSER_MESSAGE:\nActive Conversation History:\n{memory_text}\n\nUser Question: {query}",
            "overflow_telemetry": {
                "overflow_occurred": overflow_occurred,
                "limit": context_limit,
                "initial_tokens": initial_tokens,
                "final_tokens": agent_prompt_tokens,
                "steps": overflow_steps
            }
        }}
