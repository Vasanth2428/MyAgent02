import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.scraper_worker import safe_truncate_text
from src.agents.code_critic_worker import code_critic_worker_node, CriticReport, CriticFinding
from src.agents.critic_worker import critic_worker_node
from src.agents.utility_worker import utility_worker_node
from src.agents.coding_worker import coding_worker_node, tools_map


class TestHardenedWorkers(unittest.TestCase):
    def test_safe_truncate_text(self):
        """Test safe_truncate_text doesn't cut mid-sentence or mid-word when possible."""
        text = "Hello world. This is a sentence. And another one here."
        # If we request truncate at 33 chars, it should end at the period of the second sentence.
        truncated = safe_truncate_text(text, 33)
        self.assertEqual(truncated, "Hello world. This is a sentence.")

        # If we ask for something very short, it should fallback to space boundary
        text2 = "Hello world somethinglong"
        truncated2 = safe_truncate_text(text2, 13)
        self.assertEqual(truncated2, "Hello world")

    @patch("src.agents.code_critic_worker.get_critic_model")
    @patch("src.agents.coding_worker.get_retrieval_service")
    def test_code_critic_retry_limit(self, mock_retrieval, mock_get_critic):
        """Code critic should abort retry loops and proceed after reaching retry limit."""
        # Mock structured report output
        mock_report = CriticReport(
            valid=False,
            findings=[CriticFinding(issue_type="security_risk", details="Path traversal vulnerability", severity="critical")],
            criticism_summary="Security validation failed."
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_report
        mock_get_critic.return_value = mock_llm

        # Set up a state that has already hit the retry count of 2
        state = {
            "scratchpad": "Some findings",
            "current_task": "Write secure files",
            "worker_outputs": {"coding_worker": "def exploit(): pass"},
            "critic_retry_count": 2,
            "plan": ["Original task"]
        }

        res = code_critic_worker_node(state)
        # It should not append a retry task to the plan, nor append RETRY_REQUIRED
        self.assertEqual(res["critic_retry_count"], 0)
        self.assertNotIn("plan", res)
        self.assertNotIn("RETRY_REQUIRED", res["messages"][0].content)
        self.assertIn("Max validation retry limit reached", res["messages"][0].content)

        # Test when retry count is 0 (first retry)
        state_first = {
            "scratchpad": "Some findings",
            "current_task": "Write secure files",
            "worker_outputs": {"coding_worker": "def exploit(): pass"},
            "critic_retry_count": 0,
            "plan": ["Original task"]
        }
        res_first = code_critic_worker_node(state_first)
        self.assertEqual(res_first["critic_retry_count"], 1)
        self.assertTrue(any("FIX:" in p for p in res_first["plan"]))
        self.assertIn("RETRY_REQUIRED", res_first["messages"][0].content)

    @patch("src.agents.critic_worker.get_reasoning_model")
    def test_critic_retry_limit(self, mock_get_critic):
        """Critic specialist should abort retry loops when limit is reached."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Contradiction detected! RETRY_REQUIRED"
        mock_llm.invoke.return_value = mock_response
        mock_get_critic.return_value = mock_llm
    
        state = {
            "messages": [HumanMessage(content="Query")],
            "scratchpad": "Some conflicting findings",
            "current_task": "Fact check query",
            "critic_retry_count": 2,
            "plan": ["Original plan"]
        }
    
        res = critic_worker_node(state)
        # It should strip RETRY_REQUIRED and not append error task, but should suggest an alternative approach
        self.assertEqual(res["critic_retry_count"], 2)
        self.assertIn("plan", res)
        self.assertTrue(any("ALTERNATIVE APPROACH" in p for p in res["plan"]))
        self.assertNotIn("RETRY_REQUIRED", res["messages"][0].content)
        self.assertIn("Max validation retry limit reached", res["messages"][0].content)

    def test_utility_worker_coding_keyword_bypass(self):
        """Utility worker should redirect coding tasks to the coding specialist."""
        state = {
            "current_task": "Can you summarize the logic flaws in my database config and calculate time complexity?",
            "scratchpad": "",
            "messages": []
        }
        res = utility_worker_node(state)
        self.assertIn("route code/repository analysis tasks to the coding specialist", res["messages"][0].content)
        self.assertEqual(res["worker_complete"]["utility_worker"], True)

    def test_utility_worker_actual_summarization(self):
        """Utility worker should summarize actual scratchpad text if available."""
        state = {
            "current_task": "summarize",
            "scratchpad": "Line 1 of logs.\nLine 2 of findings.\nLine 3 of results.",
            "messages": []
        }
        res = utility_worker_node(state)
        # Should not return the static "please provide text" message, but actually summarize/truncate it
        self.assertNotIn("please provide the text you'd like me to summarize", res["messages"][0].content.lower())
        self.assertIn("findings", res["messages"][0].content)

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_interrupted_timeout(self, mock_get_model):
        """Coding worker should return completed=False and report interruption on loop timeout."""
        mock_tool_call = {
            "name": "list_files",
            "args": {"directory": "."},
            "id": "call_123"
        }
        
        mock_response_with_tools = MagicMock()
        mock_response_with_tools.tool_calls = [mock_tool_call]
        mock_response_with_tools.content = "Need to check files again..."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response_with_tools
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "List files repeatedly",
            "scratchpad": "",
            "messages": []
        }
        
        with patch.dict(tools_map, {"list_files": MagicMock(invoke=MagicMock(return_value="file1.txt"))}):
            res = coding_worker_node(state)
            self.assertEqual(res["worker_complete"]["coding_worker"], False)
            self.assertIn("Interrupted - execution limit reached", res["scratchpad"])
            self.assertIn("Interrupted: reached execution limit", res["messages"][0].content)


if __name__ == "__main__":
    unittest.main()
