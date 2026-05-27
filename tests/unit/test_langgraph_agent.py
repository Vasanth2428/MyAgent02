import unittest
from unittest.mock import MagicMock, patch
from core.agent import RAGAgent
from core.graph import RAGLangGraph, AgentState

class TestLangGraphAgent(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.engine.stats = {"queries": 0}
        self.engine.llm_service = MagicMock()
        self.engine.llm_service.model = "test-model"
        
        self.memory = MagicMock()
        self.memory.get_active_context.return_value = "User: Hello\nAssistant: Hi"
        self.engine.get_memory.return_value = self.memory
        
        self.agent = RAGAgent(self.engine)
        self.graph = self.agent.graph

    def test_graph_compiles(self):
        """Verify the LangGraph compiled graph is initialized correctly."""
        self.assertIsNotNone(self.graph.compiled_graph)

    def test_early_exit_check_greeting(self):
        """Test early exit check detection for greetings."""
        state = {
            "query": "hello",
            "session_id": "test",
            "context_limit": None,
            "events_queue": []
        }
        res = self.graph.early_exit_check(state)
        self.assertEqual(res["early_exit_type"], "greeting")
        self.assertTrue(any(e["state"] == "STREAMING_FINAL_RESPONSE" for e in res["events_queue"]))

    def test_early_exit_check_registry(self):
        """Test early exit check detection for registry queries."""
        state = {
            "query": "show me the sources please",
            "session_id": "test",
            "context_limit": None,
            "events_queue": []
        }
        res = self.graph.early_exit_check(state)
        self.assertEqual(res["early_exit_type"], "registry")

    def test_early_exit_check_nominal(self):
        """Test early exit check for normal queries (no early exit)."""
        state = {
            "query": "what is the CPU load?",
            "session_id": "test",
            "context_limit": None,
            "events_queue": []
        }
        res = self.graph.early_exit_check(state)
        self.assertIsNone(res["early_exit_type"])

    def test_route_early_exit(self):
        """Test route_early_exit router returns correct node name."""
        state_exit = {"early_exit_type": "greeting"}
        state_norm = {"early_exit_type": None}
        self.assertEqual(self.graph.route_early_exit(state_exit), "early_exit_execute")
        self.assertEqual(self.graph.route_early_exit(state_norm), "overflow_recovery")

    def test_route_after_tool(self):
        """Test route_after_tool routing decision based on iterations."""
        state_under = {"iteration": 2}
        state_over = {"iteration": 3}
        self.assertEqual(self.graph.route_after_tool(state_under), "reasoning")
        self.assertEqual(self.graph.route_after_tool(state_over), "synthesis")

    def test_call_llm_with_retry_success(self):
        """Test that _call_llm_with_retry succeeds when LLM behaves correctly."""
        mock_completion = MagicMock()
        self.engine.client.chat.completions.create.return_value = mock_completion
        
        res = self.graph._call_llm_with_retry([{"role": "user", "content": "hi"}], max_retries=2)
        self.assertEqual(res, mock_completion)
        self.assertEqual(self.engine.client.chat.completions.create.call_count, 1)

    @patch("time.sleep", return_value=None)
    def test_call_llm_with_retry_transient_recovery(self, mock_sleep):
        """Test _call_llm_with_retry retries on transient errors and recovers."""
        mock_completion = MagicMock()
        # Raise Exception on 1st call, succeed on 2nd
        self.engine.client.chat.completions.create.side_effect = [
            Exception("Groq API Rate Limit Exceeded (429)"),
            mock_completion
        ]
        
        res = self.graph._call_llm_with_retry([{"role": "user", "content": "hi"}], max_retries=3)
        self.assertEqual(res, mock_completion)
        self.assertEqual(self.engine.client.chat.completions.create.call_count, 2)
        self.assertTrue(mock_sleep.called)

    @patch("time.sleep", return_value=None)
    def test_call_llm_with_retry_permanent_failure(self, mock_sleep):
        """Test _call_llm_with_retry raises exception after all retries fail."""
        self.engine.client.chat.completions.create.side_effect = Exception("Service Unavailable (503)")
        
        with self.assertRaises(Exception) as context:
            self.graph._call_llm_with_retry([{"role": "user", "content": "hi"}], max_retries=3)
            
        self.assertIn("503", str(context.exception))
        self.assertEqual(self.engine.client.chat.completions.create.call_count, 3)

    def test_execute_tool_malformed_action(self):
        """Test execute_tool node handles None or malformed parsed_action gracefully without crashing."""
        state_none = {
            "parsed_action": None,
            "scratchpad": "Initial scratchpad",
            "events_queue": []
        }
        res_none = self.graph.execute_tool(state_none)
        self.assertIn("Error: Invalid or missing action definition", res_none["scratchpad"])
        self.assertTrue(any("Invalid or missing action definition" in e["output"] for e in res_none["events_queue"]))

        state_short = {
            "parsed_action": ("web_search",), # length 1 tuple
            "scratchpad": "Initial scratchpad",
            "events_queue": []
        }
        res_short = self.graph.execute_tool(state_short)
        self.assertIn("Error: Invalid or missing action definition", res_short["scratchpad"])

    def test_synthesis_fallback(self):
        """Test synthesis node falls back gracefully to markdown list from scratchpad when Groq LLM fails."""
        # Force LLM to fail
        self.engine.client.chat.completions.create.side_effect = Exception("Groq is completely down")
        
        state = {
            "query": "What is the CPU usage?",
            "scratchpad": "Thought: I need to check system stats.\nAction: get_system_stats[]\nObservation: System Stats: CPU=12%, RAM=45%\nThought: I have the info.",
            "session_id": "test_fallback",
            "memory_text": "History context",
            "overflow_occurred": False,
            "context_limit": None,
            "overflow_steps": [],
            "goals_set": ["Check system stats"],
            "actions_taken": [],
            "llm_call_count": 0,
            "events_queue": []
        }
        
        res = self.graph.synthesis(state)
        # Verify it returned a response structured with our fallback markdown
        self.assertIn("### Investigation Summary (Fallback)", res["final_response"])
        self.assertIn("Executed action", res["final_response"])
        self.assertIn("Observation result", res["final_response"])
        self.assertIn("System Stats: CPU=12%, RAM=45%", res["final_response"])
        # Verify answer chunks were streamed
        self.assertTrue(any(e["event"] == "answer_chunk" for e in res["events_queue"]))

    def test_telemetry_resilience(self):
        """Test streaming_final_answer and synthesis nodes are safe from database saves and stats lookup crashes."""
        # Cause stats lookup and database save to raise exceptions
        self.engine.stats = None # Will raise TypeError or KeyError
        self.engine.save_memory.side_effect = Exception("Database locked")
        
        state = {
            "final_response": "Here is the final answer.",
            "query": "hello",
            "session_id": "test_session",
            "memory_text": "hello",
            "overflow_occurred": False,
            "context_limit": None,
            "overflow_steps": [],
            "goals_set": [],
            "actions_taken": [],
            "llm_call_count": 1,
            "events_queue": []
        }
        
        # This call should complete successfully without raising "Database locked" or TypeErrors
        res = self.graph.streaming_final_answer(state)
        self.assertTrue(any(e["event"] == "done" for e in res["events_queue"]))

if __name__ == "__main__":
    unittest.main()
