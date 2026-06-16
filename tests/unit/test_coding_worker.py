import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from src.agents.coding_worker import coding_worker_node, tools_map, is_task_compatible, parse_malformed_tool_calls



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
            "messages": [],
            "patch_is_verified": True
        }
        
        mock_modify = MagicMock()
        with patch.dict(tools_map, {"modify_files": mock_modify}):
            res = coding_worker_node(state)
            mock_modify.invoke.assert_not_called()
            # Handle both dict and Command responses
            from langgraph.types import Command
            update = res.update if isinstance(res, Command) else res
            # The agent gets the block observation and pauses, return worker_complete as False and waiting_for_approval as True
            self.assertEqual(update["worker_complete"]["coding_worker"], False)
            self.assertEqual(update["waiting_for_approval"], True)

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
            "messages": [],
            "patch_is_verified": True
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
        from langgraph.types import Command
        update = res.update if isinstance(res, Command) else res
        self.assertEqual(update["worker_complete"]["coding_worker"], True)
        self.assertIn("strictly restricted to writing frontend code using the React framework", update["worker_outputs"]["coding_worker"])

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
        from langgraph.types import Command
        update = res.update if isinstance(res, Command) else res
        self.assertEqual(update["worker_complete"]["coding_worker"], True)
        self.assertIn("strictly restricted to writing frontend code using the React framework", update["worker_outputs"]["coding_worker"])

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_node_bypass_hitl(self, mock_get_model):
        """Coding worker should execute the tool call directly without blocking and requesting approval when bypass_hitl is True."""
        mock_tool_call = {
            "name": "create_files",
            "args": {"filepath": "bypass_test.py", "content": "print('bypass')"},
            "id": "call_bypass"
        }
        
        mock_response = MagicMock()
        mock_response.tool_calls = [mock_tool_call]
        mock_response.content = "Creating file."
        
        mock_response_stop = MagicMock()
        mock_response_stop.tool_calls = []
        mock_response_stop.content = "Finished creating file."
        
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [mock_response, mock_response_stop]
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Create bypass file",
            "scratchpad": "",
            "messages": [],
            "bypass_hitl": True
        }
        
        mock_create = MagicMock()
        mock_create.invoke = MagicMock(return_value="Success: Created file 'bypass_test.py'")
        with patch.dict(tools_map, {"create_files": mock_create}):
            res = coding_worker_node(state)
            
            # The agent should execute create_files directly, not block
            mock_create.invoke.assert_called_once_with({"filepath": "bypass_test.py", "content": "print('bypass')"})
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            self.assertNotIn("waiting_for_approval", res)

    @patch("src.agents.coding_worker.get_validation_model")
    def test_coding_worker_accepts_repository_search_task(self, mock_get_val_model):
        """is_task_compatible should accept repository search tasks."""
        mock_response = MagicMock()
        mock_response.content = '{"is_compatible": true, "explanation": ""}'
        mock_val_llm = MagicMock()
        mock_val_llm.invoke.return_value = mock_response
        mock_get_val_model.return_value = mock_val_llm

        is_compatible, explanation = is_task_compatible("Search for the login function in repository")
        self.assertTrue(is_compatible)
        self.assertEqual(explanation, "")

    @patch("src.agents.coding_worker.get_validation_model")
    def test_coding_worker_accepts_workspace_html_creation_task(self, mock_get_val_model):
        """is_task_compatible should accept workspace HTML creation tasks."""
        mock_response = MagicMock()
        mock_response.content = '{"is_compatible": true, "explanation": ""}'
        mock_val_llm = MagicMock()
        mock_val_llm.invoke.return_value = mock_response
        mock_get_val_model.return_value = mock_val_llm

        is_compatible, explanation = is_task_compatible("Create a helper hello.html inside ./workspace")
        self.assertTrue(is_compatible)
        self.assertEqual(explanation, "")

    def test_coding_worker_parses_malformed_groq_tool_call(self):
        """parse_malformed_tool_calls should parse various forms of malformed JSON / action inputs."""
        # Scenario A: JSON inside markdown code block
        content_a = """
Some text before the block.
```json
{
  "name": "create_files",
  "args": {"filepath": "test.py", "content": "print(1)"}
}
```
"""
        calls_a = parse_malformed_tool_calls(content_a)
        self.assertEqual(len(calls_a), 1)
        self.assertEqual(calls_a[0]["name"], "create_files")
        self.assertEqual(calls_a[0]["args"]["filepath"], "test.py")
        self.assertTrue("id" in calls_a[0])

        # Scenario B: JSON without code blocks, with arguments instead of args
        content_b = """
{
  "name": "modify_files",
  "arguments": {"filepath": "test.py", "target_code": "a", "replacement_code": "b"}
}
"""
        calls_b = parse_malformed_tool_calls(content_b)
        self.assertEqual(len(calls_b), 1)
        self.assertEqual(calls_b[0]["name"], "modify_files")
        self.assertEqual(calls_b[0]["args"]["filepath"], "test.py")

        # Scenario C: Action / Action Input style
        content_c = """
Thought: I need to list files.
Action: list_files
Action Input: {"directory": "."}
"""
        calls_c = parse_malformed_tool_calls(content_c)
        self.assertEqual(len(calls_c), 1)
        self.assertEqual(calls_c[0]["name"], "list_files")
        self.assertEqual(calls_c[0]["args"]["directory"], ".")

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_blocks_create_files_without_approval(self, mock_get_model):
        """Coding worker should block create_files if not approved, store it in pending approvals, and return waiting_for_approval."""
        mock_tool_call = {
            "name": "create_files",
            "args": {"filepath": "new_file.py", "content": "print('hello')"},
            "id": "call_create_123"
        }
        
        mock_response = MagicMock()
        mock_response.tool_calls = [mock_tool_call]
        mock_response.content = "Creating file."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Create new file",
            "scratchpad": "",
            "messages": [],
            "configurable": {"thread_id": "test_thread"},
            "patch_is_verified": True
        }
        
        with patch("src.graph.supervisor.is_file_approved") as mock_approved:
            mock_approved.return_value = False
            res = coding_worker_node(state)
            
        from langgraph.types import Command
        update = res.update if isinstance(res, Command) else res
        
        self.assertEqual(update["waiting_for_approval"], True)
        self.assertEqual(update["approval_filepath"], "new_file.py")
        self.assertEqual(update["approval_tool"], "create_files")
        
        # Verify stores tool_call_id
        pending = update["pending_file_approvals"]
        self.assertIn("new_file.py", pending)
        self.assertEqual(pending["new_file.py"]["tool_call_id"], "call_create_123")
        self.assertEqual(pending["new_file.py"]["tool"], "create_files")
        self.assertEqual(pending["new_file.py"]["args"]["content"], "print('hello')")

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_resumes_after_approval_without_reexecuting_tool(self, mock_get_model):
        """Coding worker should resume after approval using stored resume details, replacing the placeholder and not re-running the tool."""
        # Create private message transcript with a placeholder ToolMessage
        agent_messages = [
            SystemMessage(content="system_prompt"),
            HumanMessage(content="task"),
            AIMessage(content="I will create the file.", tool_calls=[{"name": "create_files", "args": {"filepath": "res.py", "content": "1"}, "id": "call_res"}]),
            ToolMessage(content="Approval required...", tool_call_id="call_res", name="create_files")
        ]
        
        # When model is invoked next, it shouldn't ask for tool calls again; it should finish.
        mock_response_stop = MagicMock()
        mock_response_stop.tool_calls = []
        mock_response_stop.content = "Finished task after resume."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response_stop
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Create res.py",
            "scratchpad": "",
            "messages": [],
            "coding_worker_messages": agent_messages,
            "coding_worker_step": 1,
            "coding_worker_tool_calls_count": 1,
            "coding_worker_resume_tool_result": "Success: Created res.py",
            "coding_worker_resume_tool_call_id": "call_res",
            "patch_is_verified": True
        }
        
        mock_create = MagicMock()
        with patch.dict(tools_map, {"create_files": mock_create}):
            res = coding_worker_node(state)
            
            # create_files should NOT be executed again
            mock_create.invoke.assert_not_called()
            
            self.assertEqual(res["worker_complete"]["coding_worker"], True)
            self.assertEqual(res["worker_outputs"]["coding_worker"], "Finished task after resume.")
            
            # verify resume result was appended to the private transcript
            invoked_messages = mock_llm.invoke.call_args[0][0]
            self.assertEqual(len(invoked_messages), 6)
            self.assertEqual(invoked_messages[3].content, "Approval required...")
            self.assertEqual(invoked_messages[4].content, "Human Approved. Execution Result:\nSuccess: Created res.py")

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_clears_private_resume_state_after_completion(self, mock_get_model):
        """Coding worker should clear private resume state from state dictionary after task is successfully completed."""
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Task finished cleanly."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Simple task",
            "scratchpad": "",
            "messages": [],
            "coding_worker_messages": [HumanMessage(content="task")],
            "coding_worker_step": 2,
            "coding_worker_tool_calls_count": 2,
            "coding_worker_resume_tool_result": "Result",
            "coding_worker_resume_tool_call_id": "call_123"
        }
        
        res = coding_worker_node(state)
        
        self.assertEqual(res["worker_complete"]["coding_worker"], True)
        self.assertEqual(res["coding_worker_messages"], [])
        self.assertEqual(res["coding_worker_step"], 0)
        self.assertEqual(res["coding_worker_tool_calls_count"], 0)
        self.assertIsNone(res["coding_worker_resume_tool_result"])
        self.assertIsNone(res["coding_worker_resume_tool_call_id"])

    def test_scaffold_react_app_tool(self):
        """Verifies scaffold_react_app tool executes correctly."""
        from src.agents.coding_worker import scaffold_react_app
        with patch("src.agents.coding_worker._scaffold_react_app") as mock_scaffold:
            mock_scaffold.return_value = "Success: Scaffolded React application 'test_proj' successfully."
            res = scaffold_react_app.invoke({"project_name": "test_proj"})
            self.assertEqual(res, "Success: Scaffolded React application 'test_proj' successfully.")
            mock_scaffold.assert_called_once_with("test_proj")

    @patch("src.agents.coding_worker.get_coding_model")
    def test_coding_worker_multi_file_approval_queue(self, mock_get_model):
        """Verifies that multiple file operations are queued and processed together."""
        from src.agents.coding_worker import _pending_approvals, _session_resume_results, execute_pending_approval
        
        # Clear caches
        session_id = "test_multi_session"
        if session_id in _pending_approvals:
            del _pending_approvals[session_id]
        if session_id in _session_resume_results:
            del _session_resume_results[session_id]
            
        # 1. Test queue execution
        mock_create = MagicMock()
        mock_create.invoke.side_effect = lambda args: f"Created {args['filepath']}"
        
        with patch.dict(tools_map, {"create_files": mock_create}):
            from src.agents.coding_worker import set_pending_approval
            set_pending_approval(session_id, "file1.py", "create_files", {"filepath": "file1.py", "content": "c1"}, "id1")
            set_pending_approval(session_id, "file2.py", "create_files", {"filepath": "file2.py", "content": "c2"}, "id2")
            
            res_exec = execute_pending_approval(session_id)
            self.assertIn("file1.py", res_exec)
            self.assertIn("file2.py", res_exec)
            
            # verify results cache
            self.assertIn(session_id, _session_resume_results)
            self.assertEqual(len(_session_resume_results[session_id]), 2)
            self.assertEqual(_session_resume_results[session_id][0]["result"], "Created file1.py")
            self.assertEqual(_session_resume_results[session_id][1]["result"], "Created file2.py")
            
        # 2. Test resume from multiple results in coding_worker_node
        agent_messages = [
            SystemMessage(content="system_prompt"),
            HumanMessage(content="task"),
            AIMessage(content="I will create two files.", tool_calls=[
                {"name": "create_files", "args": {"filepath": "file1.py", "content": "c1"}, "id": "id1"},
                {"name": "create_files", "args": {"filepath": "file2.py", "content": "c2"}, "id": "id2"}
            ]),
            ToolMessage(content="Approval required...", tool_call_id="id1", name="create_files"),
            ToolMessage(content="Approval required...", tool_call_id="id2", name="create_files")
        ]
        
        mock_response_stop = MagicMock()
        mock_response_stop.tool_calls = []
        mock_response_stop.content = "Finished multi-resume."
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response_stop
        mock_get_model.return_value = mock_llm
        
        state = {
            "current_task": "Create files",
            "scratchpad": "",
            "messages": [],
            "configurable": {"thread_id": session_id},
            "coding_worker_messages": agent_messages,
            "coding_worker_step": 1,
            "coding_worker_tool_calls_count": 2,
            "patch_is_verified": True
        }
        
        # We must re-populate the cache because execute_pending_approval deleted it and put it in _session_resume_results
        _session_resume_results[session_id] = [
            {"tool_call_id": "id1", "tool_name": "create_files", "result": "Success: Created file1.py"},
            {"tool_call_id": "id2", "tool_name": "create_files", "result": "Success: Created file2.py"}
        ]
        
        res_node = coding_worker_node(state)
        self.assertEqual(res_node["worker_complete"]["coding_worker"], True)
        
        invoked_messages = mock_llm.invoke.call_args[0][0]
        # Should have System, Human, AI, Tool1_Placeholder, Tool2_Placeholder, Tool1_Result, Tool2_Result, Response
        self.assertEqual(len(invoked_messages), 8)
        self.assertEqual(invoked_messages[5].content, "Human Approved. Execution Result:\nSuccess: Created file1.py")
        self.assertEqual(invoked_messages[6].content, "Human Approved. Execution Result:\nSuccess: Created file2.py")


if __name__ == "__main__":
    unittest.main()


