import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage
from src.agents.coding_worker import coding_worker_node, tools_map


class TestCodingWorker(unittest.TestCase):
    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_missing_task(self, mock_get_model):
        """Coding worker should exit gracefully if no task is provided."""
        state = {
            "current_task": "",
            "scratchpad": "",
            "messages": []
        }
        res = coding_worker_node(state)
        self.assertEqual(res["worker_complete"]["coding_worker"], True)
        self.assertIn("No instruction provided", res["worker_outputs"]["coding_worker"])

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_loop_termination(self, mock_get_model):
        """Coding worker should execute the model and terminate the loop when no tools are called."""
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Finished coding task successfully."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Write hello world script",
            "scratchpad": "Blackboard info",
            "messages": []
        }
        
        res = coding_worker_node(state)
        self.assertEqual(res["worker_complete"]["coding_worker"], True)
        self.assertEqual(res["worker_outputs"]["coding_worker"], "Finished coding task successfully.")
        self.assertIn("Coding Worker", res["scratchpad"])
        self.assertEqual(res["next_agent"], "supervisor")

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_tool_limit(self, mock_get_model):
        """Coding worker should stop calling tools if the max_tool_calls limit (15) is reached."""
        # Create a mock response that keeps requesting tool calls
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
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            self.assertIn("Coding Worker", res["scratchpad"])

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_delete_file(self, mock_get_model):
        """Coding worker should execute the delete_file tool when requested by the model."""
        mock_tool_call = {
            "name": "delete_file",
            "args": {"filepath": "temp_delete.py"},
            "id": "call_999"
        }
        
        mock_response = MagicMock()
        mock_response.tool_calls = [mock_tool_call]
        mock_response.content = "Deleting file."
        
        mock_response_stop = MagicMock()
        mock_response_stop.tool_calls = []
        mock_response_stop.content = "Finished deleting."
        
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response, mock_response_stop]
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Delete the temp file",
            "scratchpad": "",
            "messages": []
        }
        
        mock_delete = MagicMock()
        mock_delete.invoke = MagicMock(return_value="Success: Deleted file 'temp_delete.py'")
        with patch.dict(tools_map, {"delete_file": mock_delete}):
            res = coding_worker_node(state)
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            mock_delete.invoke.assert_called_once_with({"filepath": "temp_delete.py"})


if __name__ == "__main__":
    unittest.main()

