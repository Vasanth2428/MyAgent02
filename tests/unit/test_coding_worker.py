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
            self.assertEqual(res["worker_complete"]["coding_worker"], False)
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
            "scratchpad": "[APPROVED: temp_delete.py]",
            "messages": []
        }
    
        mock_delete = MagicMock()
        mock_delete.invoke = MagicMock(return_value="Success: Deleted file 'temp_delete.py'")
        with patch.dict(tools_map, {"delete_file": mock_delete}):
            with patch("src.graph.supervisor.is_file_approved") as mock_approved:
                mock_approved.return_value = True
                res = coding_worker_node(state)
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            mock_delete.invoke.assert_called_once_with({"filepath": "temp_delete.py"})

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_blocked_without_approval(self, mock_get_model):
        """Coding worker should block modify_files if not approved in scratchpad."""
        mock_tool_call = {
            "name": "modify_files",
            "args": {"filepath": "important.py", "target_code": "a=1", "replacement_code": "a=2"},
            "id": "call_888"
        }
        
        mock_response = MagicMock()
        mock_response.tool_calls = [mock_tool_call]
        mock_response.content = "Modifying file."
        
        mock_response_stop = MagicMock()
        mock_response_stop.tool_calls = []
        mock_response_stop.content = "Cannot proceed without approval."
        
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response, mock_response_stop]
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Modify the file",
            "scratchpad": "",
            "messages": []
        }
        
        mock_modify = MagicMock()
        with patch.dict(tools_map, {"modify_files": mock_modify}):
            res = coding_worker_node(state)
            mock_modify.invoke.assert_not_called()
            # The agent gets the block observation and pauses, return worker_complete as False and waiting_for_approval as True
            self.assertEqual(res["worker_complete"]["coding_worker"], False)
            self.assertEqual(res["waiting_for_approval"], True)

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_path_normalized_approval(self, mock_get_model):
        """Coding worker should execute create_files if approved using a normalized relative path representation."""
        mock_tool_call = {
            "name": "create_files",
            "args": {"filepath": "banking_form.html", "content": "test"},
            "id": "call_1234"
        }
    
        mock_response = MagicMock()
        mock_response.tool_calls = [mock_tool_call]
        mock_response.content = "Creating banking form."
    
        mock_response_stop = MagicMock()
        mock_response_stop.tool_calls = []
        mock_response_stop.content = "Finished creating."
    
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response, mock_response_stop]
        mock_get_model.return_value = mock_llm
    
        state = {
            "current_task": "Create banking_form.html",
            "scratchpad": "[APPROVED: ./workspace/banking_form.html]",
            "messages": []
        }
    
        mock_create = MagicMock()
        mock_create.invoke = MagicMock(return_value="Success: Created file 'banking_form.html'")
        with patch.dict(tools_map, {"create_files": mock_create}):
            with patch("src.graph.supervisor.is_file_approved") as mock_approved:
                mock_approved.return_value = True
                res = coding_worker_node(state)
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            mock_create.invoke.assert_called_once_with({"filepath": "banking_form.html", "content": "test"})

    @patch("src.agents.coding_worker.get_validation_model")
    def test_coding_worker_node_compatible_react(self, mock_get_val_model):
        """Coding worker should accept React frontend tasks."""
        mock_response = MagicMock()
        mock_response.content = '{"is_compatible": true, "explanation": ""}'
        mock_val_llm = MagicMock()
        mock_val_llm.invoke.return_value = mock_response
        mock_get_val_model.return_value = mock_val_llm

        with patch("src.agents.coding_worker.get_coding_model") as mock_get_model:
            mock_coding_response = MagicMock()
            mock_coding_response.tool_calls = []
            mock_coding_response.content = "React task done."
            mock_coding_llm = MagicMock()
            mock_coding_llm.invoke.return_value = mock_coding_response
            mock_get_model.return_value = mock_coding_llm

            state = {
                "current_task": "Create a React login form component",
                "scratchpad": "",
                "messages": []
            }
            res = coding_worker_node(state)
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            self.assertIn("React task done", res["worker_outputs"]["coding_worker"])

    @patch("src.agents.coding_worker.get_validation_model")
    def test_coding_worker_node_incompatible_vue(self, mock_get_val_model):
        """Coding worker should gracefully reject Vue frontend tasks."""
        mock_response = MagicMock()
        mock_response.content = '{"is_compatible": false, "explanation": "I apologize, but I am strictly restricted to writing frontend code using the React framework and backend code in Python."}'
        mock_val_llm = MagicMock()
        mock_val_llm.invoke.return_value = mock_response
        mock_get_val_model.return_value = mock_val_llm

        state = {
            "current_task": "Create a Vue login component",
            "scratchpad": "",
            "messages": []
        }
        res = coding_worker_node(state)
        self.assertEqual(res["worker_complete"]["coding_worker"], True)
        self.assertIn("strictly restricted to writing frontend code using the React framework", res["worker_outputs"]["coding_worker"])

    @patch("src.agents.coding_worker.get_validation_model")
    def test_coding_worker_node_incompatible_nodejs(self, mock_get_val_model):
        """Coding worker should gracefully reject Node.js backend tasks."""
        mock_response = MagicMock()
        mock_response.content = '{"is_compatible": false, "explanation": "I apologize, but I am strictly restricted to writing frontend code using the React framework and backend code in Python."}'
        mock_val_llm = MagicMock()
        mock_val_llm.invoke.return_value = mock_response
        mock_get_val_model.return_value = mock_val_llm

        state = {
            "current_task": "Build a Node.js express API endpoint",
            "scratchpad": "",
            "messages": []
        }
        res = coding_worker_node(state)
        self.assertEqual(res["worker_complete"]["coding_worker"], True)
        self.assertIn("strictly restricted to writing frontend code using the React framework", res["worker_outputs"]["coding_worker"])


if __name__ == "__main__":
    unittest.main()
