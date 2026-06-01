import unittest
from unittest.mock import MagicMock, patch
import asyncio

from core.agent import RAGAgent
from core.graph import RAGLangGraph, AgentState

class TestLangGraphAgent(unittest.IsolatedAsyncioTestCase):
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

    async def test_early_exit_check_greeting(self):
        """Test early exit check detection for greetings."""
        state = {
            "query": "hello",
            "session_id": "test",
            "context_limit": None,
            "events_queue": []
        }
        res = await self.graph.early_exit_check(state)
        self.assertEqual(res["early_exit_type"], "greeting")
        self.assertTrue(any(e["state"] == "STREAMING_FINAL_RESPONSE" for e in res["events_queue"]))

    async def test_early_exit_check_registry(self):
        """Test early exit check detection for registry queries."""
        state = {
            "query": "show me the sources please",
            "session_id": "test",
            "context_limit": None,
            "events_queue": []
        }
        res = await self.graph.early_exit_check(state)
        self.assertEqual(res["early_exit_type"], "registry")

    async def test_early_exit_check_nominal(self):
        """Test early exit check for normal queries (no early exit)."""
        state = {
            "query": "what is the CPU load?",
            "session_id": "test",
            "context_limit": None,
            "events_queue": []
        }
        res = await self.graph.early_exit_check(state)
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

    async def test_execute_tool_malformed_action(self):
        """Test execute_tool node handles None or malformed parsed_action gracefully without crashing."""
        state_none = {
            "parsed_action": None,
            "scratchpad": "Initial scratchpad",
            "events_queue": [],
            "memory_text": "",
            "actions_taken": [],
            "iteration": 1,
            "search_cache": {}
        }
        res_none = await self.graph.execute_tool(state_none)
        self.assertIn("Error: Invalid or missing action definition", res_none["scratchpad"])
        self.assertTrue(any("Invalid or missing action definition" in e["output"] for e in res_none["events_queue"]))

        state_short = {
            "parsed_action": ("web_search",), # length 1 tuple
            "scratchpad": "Initial scratchpad",
            "events_queue": [],
            "memory_text": "",
            "actions_taken": [],
            "iteration": 1,
            "search_cache": {}
        }
        res_short = await self.graph.execute_tool(state_short)
        self.assertIn("Error: Invalid or missing action definition", res_short["scratchpad"])

    async def test_synthesis_fallback(self):
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
        
        res = await self.graph.synthesis(state)
        # Verify it returned a response structured with our fallback markdown
        self.assertIn("### Investigation Summary (Fallback)", res["final_response"])
        self.assertIn("Executed action", res["final_response"])
        self.assertIn("Observation result", res["final_response"])
        self.assertIn("System Stats: CPU=12%, RAM=45%", res["final_response"])
        # Verify answer chunks were streamed
        self.assertTrue(any(e["event"] == "answer_chunk" for e in res["events_queue"]))

    async def test_telemetry_resilience(self):
        """Test streaming_final_answer and synthesis nodes are safe from database saves and stats lookup crashes."""
        # Cause stats lookup and database save to raise exceptions
        self.engine.stats = {"queries": 0}
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
            "events_queue": [],
            "initial_tokens": 0,
            "final_tokens": 0
        }
        
        # This call should complete successfully without raising "Database locked" or TypeErrors
        res = await self.graph.streaming_final_answer(state)
        self.assertTrue(any(e["event"] == "done" for e in res["events_queue"]))

    @patch("core.scraper.requests.get")
    def test_scrape_web_page_success(self, mock_get):
        """Test successful fetch and parse of a web page."""
        mock_response = MagicMock()
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.text = (
            "<html><head><title>Ignored Title</title><style>.css {color: red;}</style></head>"
            "<body><h1>Important Header</h1><p>Some paragraph text here.</p>"
            "<script>alert(1);</script></body></html>"
        )
        mock_get.return_value = mock_response
        
        from core.scraper import scrape_web_page
        text = scrape_web_page("http://example.com/test")
        self.assertIn("Important Header", text)
        self.assertIn("Some paragraph text here.", text)
        self.assertNotIn("Ignored Title", text)
        self.assertNotIn("alert(1)", text)

    @patch("core.scraper.requests.get")
    def test_scrape_web_page_http_error(self, mock_get):
        """Test that HTTP errors are gracefully captured."""
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        from core.scraper import scrape_web_page
        text = scrape_web_page("http://example.com/missing")
        self.assertIn("Error: HTTP request failed with status: 404.", text)

    @patch("core.scraper.requests.get")
    def test_scrape_web_page_timeout(self, mock_get):
        """Test that timeouts are handled gracefully."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
        
        from core.scraper import scrape_web_page
        text = scrape_web_page("http://example.com/timeout")
        self.assertIn("timed out", text)

    @patch("core.graph.scrape_web_page")
    async def test_execute_tool_web_scrape(self, mock_scrape):
        """Test that execute_tool handles the web_scrape tool, runs compressor, and caches outcome."""
        mock_scrape.return_value = (
            "This is a long web page content describing Python and RAG development.\n"
            "Here is another paragraph with specific details about LangGraph framework."
        )
        self.engine.compressor.compress.return_value = "Compressed: LangGraph framework"
        
        state = {
            "parsed_action": ("web_scrape", "https://python.org"),
            "scratchpad": "",
            "query": "What is LangGraph?",
            "session_id": "test_scrape",
            "memory_text": "History",
            "iteration": 1,
            "actions_taken": [],
            "search_cache": {},
            "events_queue": []
        }
        
        res = await self.graph.execute_tool(state)
        # Check cache was populated
        self.assertIn("https://python.org", res["search_cache"])
        self.assertEqual(res["search_cache"]["https://python.org"], "Compressed: LangGraph framework")
        # Check scratchpad updated
        self.assertIn("Compressed: LangGraph framework", res["scratchpad"])
        # Check actions_taken updated
        self.assertEqual(len(res["actions_taken"]), 1)
        self.assertEqual(res["actions_taken"][0]["tool"], "web_scrape")

    def test_get_current_time(self):
        """Test get_current_time returns a valid formatted date-time string."""
        from core.tools import get_current_time
        now_str = get_current_time()
        self.assertEqual(len(now_str), 19)
        # Check standard format YYYY-MM-DD HH:MM:SS
        import re
        self.assertTrue(re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", now_str))

    def test_secure_evaluator_valid(self):
        """Test SecureEvaluator evaluates valid mathematical operations successfully."""
        from core.tools import evaluate_math
        self.assertEqual(evaluate_math("2 + 2"), "4")
        self.assertEqual(evaluate_math("10 * 3 - 5"), "25")
        self.assertEqual(evaluate_math("(4 + 8) / 3"), "4")
        self.assertEqual(evaluate_math("2 ** 5"), "32")
        self.assertEqual(evaluate_math("-10 + 5"), "-5")
        self.assertEqual(evaluate_math("10 // 3"), "3")
        self.assertEqual(evaluate_math("10 % 3"), "1")

    def test_secure_evaluator_invalid_escapes(self):
        """Test SecureEvaluator rejects dangerous nodes like Call, Attribute, Name, etc."""
        from core.tools import evaluate_math
        # Rejects variable names/names
        self.assertTrue(evaluate_math("x + 1").startswith("Error:"))
        # Rejects function calls
        self.assertTrue(evaluate_math("abs(-5)").startswith("Error:"))
        # Rejects attribute access / commands
        self.assertTrue(evaluate_math("os.system('dir')").startswith("Error:"))
        # Rejects empty values
        self.assertTrue(evaluate_math("").startswith("Error:"))
        # Rejects exponent lockup attempts
        self.assertTrue(evaluate_math("2 ** 99999").startswith("Error:"))

    async def test_execute_tool_datetime_and_calculator(self):
        """Test execute_tool node executes get_current_time and calculator correctly."""
        state_time = {
            "parsed_action": ("get_current_time", ""),
            "scratchpad": "",
            "iteration": 1,
            "actions_taken": [],
            "events_queue": [],
            "memory_text": "History context",
            "search_cache": {}
        }
        res_time = await self.graph.execute_tool(state_time)
        self.assertIn("Current Datetime:", res_time["scratchpad"])

        state_calc = {
            "parsed_action": ("calculator", "3 * (5 + 5)"),
            "scratchpad": "",
            "iteration": 1,
            "actions_taken": [],
            "events_queue": [],
            "memory_text": "History context",
            "search_cache": {}
        }
        res_calc = await self.graph.execute_tool(state_calc)
        self.assertIn("30", res_calc["scratchpad"])

if __name__ == "__main__":
    unittest.main()
