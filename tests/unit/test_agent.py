import unittest
from unittest.mock import MagicMock, patch
from core.agent import RAGAgent

def make_mock_stream(text: str):
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = text
    return [mock_chunk]

class TestRAGAgent(unittest.TestCase):

    def setUp(self):
        # Create a mock engine
        self.engine = MagicMock()
        self.engine.stats = {"queries": 0}
        self.engine.llm_service = MagicMock()
        self.engine.llm_service.model = "test-model"
        
        # Mock memory
        self.memory = MagicMock()
        self.memory.get_active_context.return_value = "User: Hello\nAssistant: Hi"
        self.engine.get_memory.return_value = self.memory
        
        self.agent = RAGAgent(self.engine)

    def test_parse_action_valid(self):
        """Tests parsing valid ReAct tool actions."""
        text = "Thought: I need system details.\nAction: get_system_stats[]\nObservation:"
        action = self.agent.parse_action(text)
        self.assertEqual(action, ("get_system_stats", ""))

        text = "Thought: Let's query.\nAction: search_knowledge_base[what is the database password]"
        action = self.agent.parse_action(text)
        self.assertEqual(action, ("search_knowledge_base", "what is the database password"))

    def test_parse_action_invalid(self):
        """Tests parsing invalid or missing tool actions."""
        text = "Thought: I should just respond.\nFinal Answer: Hello user."
        action = self.agent.parse_action(text)
        self.assertIsNone(action)

    def test_early_exit_greeting(self):
        """Tests that greetings exit immediately without invoking LLM completions."""
        events = list(self.agent.run_stream("hello", session_id="test_sess"))
        
        # Verify first event is the thought detailing early exit
        self.assertEqual(events[0]["event"], "thought")
        self.assertIn("simple query or greeting", events[0]["text"])
        
        # Verify answer chunk event
        self.assertEqual(events[1]["event"], "answer_chunk")
        self.assertIn("Hello! How can I help you", events[1]["text"])

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_react_loop_get_stats(self, mock_vm, mock_cpu):
        """Tests a ReAct loop executing get_system_stats."""
        mock_cpu.return_value = 12.5
        mock_vm.return_value.percent = 45.0
        self.engine.retriever.get_count.return_value = 100
        
        # Prepare LLM responses: First step returns Action: get_system_stats, Second step returns Final Answer
        self.engine.client.chat.completions.create.side_effect = [
            make_mock_stream("Thought: I need system status.\nAction: get_system_stats[]"),
            make_mock_stream("Thought: I have the statistics.\nFinal Answer: The CPU is at 12.5%.")
        ]
        
        events = list(self.agent.run_stream("what are system stats", session_id="test_sess"))
        
        # Expect events:
        # 1. Thought: I need system status
        # 2. Action: get_system_stats
        # 3. Observation: System Stats:...
        # 4. Answer chunk: The CPU is at 12.5%
        # 5. Done
        
        event_types = [e["event"] for e in events]
        self.assertIn("thought", event_types)
        self.assertIn("action", event_types)
        self.assertIn("observation", event_types)
        self.assertIn("answer_chunk", event_types)
        self.assertIn("done", event_types)
        
        # Verify get_system_stats observation content
        observation_event = next(e for e in events if e["event"] == "observation")
        self.assertIn("CPU=12.5%", observation_event["output"])
        self.assertIn("Total Indexed Documents=100", observation_event["output"])

    def test_react_loop_search_knowledge_cache(self):
        """Tests that duplicate searches hit the local query cache."""
        self.engine._phase_expand.return_value = ["test query"]
        self.engine._phase_retrieve.return_value = [{"text": "document secret", "score": 0.9, "source": "docs"}]
        self.engine.compressor.compress.return_value = "compressed secret"
        
        # Response 1: Action: search
        # Response 2: Action: search again with same arg
        # Response 3: Final Answer
        self.engine.client.chat.completions.create.side_effect = [
            make_mock_stream("Thought: Searching database.\nAction: search_knowledge_base[secret_key]"),
            make_mock_stream("Thought: Searching again.\nAction: search_knowledge_base[secret_key]"),
            make_mock_stream("Thought: Done.\nFinal Answer: Password is secret.")
        ]
        
        events = list(self.agent.run_stream("what is the secret password", session_id="test_sess"))
        
        # Verify search was only executed once (check engine retrieval calls)
        self.assertEqual(self.engine._phase_retrieve.call_count, 1)

    def test_react_loop_exhaustion_fallback(self):
        """Tests that the agent falls back to final synthesis when loop iterations are exhausted."""
        self.engine.retriever.get_count.return_value = 50
        self.engine._phase_expand.return_value = ["query"]
        self.engine._phase_retrieve.return_value = [{"text": "doc content", "score": 0.8, "source": "docs"}]
        self.engine.compressor.compress.return_value = "compressed doc content"

        # Mock completions to return 3 Action steps, forcing exhaustion, and a 4th call for synthesis
        self.engine.client.chat.completions.create.side_effect = [
            make_mock_stream("Thought: Need to look up stats.\nAction: get_system_stats[]"),
            make_mock_stream("Thought: Need to search.\nAction: search_knowledge_base[query]"),
            make_mock_stream("Thought: Still thinking.\nAction: get_system_stats[]"),
            make_mock_stream("Based on my checks, here is the compiled response.")
        ]
        
        events = list(self.agent.run_stream("check everything now", session_id="test_sess"))
        
        # Verify the "Iteration limit reached. Synthesizing..." thought was yielded
        synthesis_thought = next((e for e in events if e["event"] == "thought" and "Iteration limit" in e["text"]), None)
        self.assertIsNotNone(synthesis_thought)
        
        # Verify final response contains the synthesis output
        done_event = next(e for e in events if e["event"] == "done")
        self.assertEqual(done_event["response"], "Based on my checks, here is the compiled response.")

    def test_parse_action_tolerant(self):
        """Verify the parser tolerates spaces, quotes, and omitted brackets."""
        # Double quotes
        action = self.agent.parse_action('Action: search_knowledge_base["some query"]')
        self.assertEqual(action, ("search_knowledge_base", "some query"))

        # Single quotes
        action = self.agent.parse_action("Action: search_knowledge_base['some query']")
        self.assertEqual(action, ("search_knowledge_base", "some query"))

        # Backticks
        action = self.agent.parse_action("Action: search_knowledge_base[`some query`]")
        self.assertEqual(action, ("search_knowledge_base", "some query"))

        # Extra spacing inside brackets and around action
        action = self.agent.parse_action("Action:   search_knowledge_base  [  some query  ]")
        self.assertEqual(action, ("search_knowledge_base", "some query"))

        # Omitted brackets
        action = self.agent.parse_action("Action: get_system_stats")
        self.assertEqual(action, ("get_system_stats", ""))

        # Omitted brackets with spaces
        action = self.agent.parse_action("Action: get_system_stats   ")
        self.assertEqual(action, ("get_system_stats", ""))

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_react_loop_self_correct(self, mock_vm, mock_cpu):
        """Verify that agent detects malformed formatting and self-corrects using feedback."""
        mock_cpu.return_value = 10.0
        mock_vm.return_value.percent = 30.0
        self.engine.retriever.get_count.return_value = 5
        
        # side_effect:
        # 1. Malformed response (no Action or Final Answer prefix)
        # 2. ReAct Action: get_system_stats[]
        # 3. Final Answer
        self.engine.client.chat.completions.create.side_effect = [
            make_mock_stream("Thought: Let's do something without properly formatting it."),
            make_mock_stream("Thought: Oops, I should use the correct format.\nAction: get_system_stats[]"),
            make_mock_stream("Thought: I have the stats.\nFinal Answer: System CPU is 10.0%.")
        ]
        
        events = list(self.agent.run_stream("tell me system stats", session_id="test_sess"))
        
        event_types = [e["event"] for e in events]
        self.assertIn("observation", event_types)
        
        # Check that we received the formatting error observation
        observation_events = [e for e in events if e["event"] == "observation"]
        self.assertTrue(any("Error: Your response did not contain a valid ReAct Action" in obs["output"] for obs in observation_events))
        
        # Verify that get_system_stats was actually executed afterwards
        action_events = [e for e in events if e["event"] == "action"]
        self.assertTrue(any(act["tool"] == "get_system_stats" for act in action_events))
        
        # Verify the final answer
        done_event = next(e for e in events if e["event"] == "done")
        self.assertEqual(done_event["response"], "System CPU is 10.0%.")


if __name__ == "__main__":
    unittest.main()

