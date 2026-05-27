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

if __name__ == "__main__":
    unittest.main()
