"""
Integration tests for the Structured State Pipeline.

Validates that the multi-agent system correctly uses structured AgentState
variables (not raw scratchpad text) as the source of truth for:
  1. Sequential context aggregation via scratchpad_references
  2. Human-in-the-Loop routing via waiting_for_approval / pending_file_approvals
  3. Loop detection via worker_output_ids
  4. Multi-hop context preservation across aggregation cycles
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.graph.workflow import aggregate_parallel_results_node
from src.graph.supervisor import supervisor_node, SupervisorDecision


# ---------------------------------------------------------------------------
# 1. Aggregation Node: Scratchpad Buffering
# ---------------------------------------------------------------------------

class TestAggregationBuffering:
    """Verify aggregate_parallel_results_node transitions volatile scratchpad
    text into the persistent scratchpad_references list before clearing."""

    def test_scratchpad_text_buffered_into_references(self):
        """Active scratchpad content should be archived into scratchpad_references."""
        state = {
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "scratchpad_references": [],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {},
            "scratchpad": "  Found revenue data in Q3 report.  ",
        }

        result = aggregate_parallel_results_node(state)

        assert result["scratchpad"] == "", "Scratchpad should be cleared after aggregation"
        assert any("Found revenue data in Q3 report" in ref for ref in result["scratchpad_references"]), \
            "Scratchpad text should be buffered into scratchpad_references"

    def test_worker_outputs_cached_and_summarized(self):
        """Each worker output should be stored in the cache and a summary reference appended."""
        state = {
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "scratchpad_references": [],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {
                "rag_worker": "Company X revenue is $10M in FY2024 according to the annual report.",
                "web_worker": "Current stock price of Company X is $45.20 as of today.",
            },
            "scratchpad": "",
        }

        result = aggregate_parallel_results_node(state)

        # Both workers should have cache IDs
        assert "rag_worker" in result["worker_output_ids"]
        assert "web_worker" in result["worker_output_ids"]

        # Both workers should have summaries
        assert "rag_worker" in result["worker_output_summaries"]
        assert "web_worker" in result["worker_output_summaries"]

        # Scratchpad references should contain formatted summaries
        refs_text = "\n".join(result["scratchpad_references"])
        assert "Rag Worker" in refs_text
        assert "Web Worker" in refs_text

        # Worker outputs dict should be cleared
        assert result["worker_outputs"] == {}

    def test_empty_scratchpad_not_buffered(self):
        """Blank or whitespace-only scratchpad should not produce a reference entry."""
        state = {
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "scratchpad_references": ["existing reference"],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {},
            "scratchpad": "   ",
        }

        result = aggregate_parallel_results_node(state)

        # Only the pre-existing reference should remain
        assert len(result["scratchpad_references"]) == 1
        assert result["scratchpad_references"][0] == "existing reference"

    def test_empty_worker_output_skipped(self):
        """Worker outputs that are empty strings should be skipped."""
        state = {
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "scratchpad_references": [],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {"rag_worker": "", "web_worker": "Valid output here"},
            "scratchpad": "",
        }

        result = aggregate_parallel_results_node(state)

        # Only web_worker should have a cache entry
        assert "rag_worker" not in result["worker_output_ids"]
        assert "web_worker" in result["worker_output_ids"]

    def test_next_agent_set_to_supervisor(self):
        """After aggregation, control should always return to supervisor."""
        state = {
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "scratchpad_references": [],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {},
            "scratchpad": "",
        }

        result = aggregate_parallel_results_node(state)
        assert result["next_agent"] == "supervisor"

    def test_preserves_existing_state_fields(self):
        """Pre-existing active_document_ids, task_hashes, file_status_flags should pass through."""
        state = {
            "worker_output_ids": {"prev_worker": "cache_123"},
            "worker_output_summaries": {"prev_worker": "old summary"},
            "scratchpad_references": ["old ref"],
            "active_document_ids": ["doc_1", "doc_2"],
            "task_hashes": ["hash_a"],
            "file_status_flags": {"file.py": "modified"},
            "worker_outputs": {},
            "scratchpad": "",
        }

        result = aggregate_parallel_results_node(state)

        assert result["active_document_ids"] == ["doc_1", "doc_2"]
        assert result["task_hashes"] == ["hash_a"]
        assert result["file_status_flags"] == {"file.py": "modified"}
        # Previous worker cache should be preserved
        assert result["worker_output_ids"]["prev_worker"] == "cache_123"


# ---------------------------------------------------------------------------
# 2. HITL Structured Routing
# ---------------------------------------------------------------------------

class TestHITLStructuredRouting:
    """Verify supervisor reads structured flags for Human-in-the-Loop,
    NOT regex-parsed scratchpad text."""

    def _base_hitl_state(self, user_message="approve", waiting=True, pending=None):
        """Helper to build a standard HITL-blocked state."""
        return {
            "messages": [
                HumanMessage(content="Create a helper file"),
                AIMessage(content="File creation blocked pending approval."),
                HumanMessage(content=user_message),
            ],
            "plan": ["Create file"],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 8,
            "waiting_for_approval": waiting,
            "pending_file_approvals": pending or {"workspace/utils.py": {"tool": "create_files"}},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

    @patch("src.graph.supervisor.get_routing_model")
    @patch("src.agents.coding_worker.execute_pending_approval", return_value="Changes applied successfully")
    @patch("src.agents.coding_worker.clear_pending_approval")
    @patch("src.tools.coding_tools._get_absolute_path", side_effect=lambda p: p)
    def test_approval_clears_structured_flags(self, mock_path, mock_clear, mock_exec, mock_get_model):
        """After approval, waiting_for_approval=False and pending_file_approvals={}."""
        mock_decision = SupervisorDecision(
            plan=["Create file"], next_agent="synthesizer", current_task=""
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = self._base_hitl_state(user_message="Yes, go ahead")
        result = supervisor_node(state)

        assert result.get("waiting_for_approval") is False
        assert result.get("pending_file_approvals") == {}

    @patch("src.graph.supervisor.get_routing_model")
    @patch("src.agents.coding_worker.execute_pending_approval", return_value="Applied")
    @patch("src.agents.coding_worker.clear_pending_approval")
    @patch("src.tools.coding_tools._get_absolute_path", side_effect=lambda p: p)
    def test_approval_writes_to_references_not_scratchpad(self, mock_path, mock_clear, mock_exec, mock_get_model):
        """HITL approval note should appear in scratchpad_references, not scratchpad."""
        mock_decision = SupervisorDecision(
            plan=["Create file"], next_agent="synthesizer", current_task=""
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = self._base_hitl_state(user_message="approve it")
        result = supervisor_node(state)

        refs = result.get("scratchpad_references", [])
        assert any("[SYSTEM HITL]: User approved modifications" in r for r in refs)

    @patch("src.graph.supervisor.get_routing_model")
    @patch("src.agents.coding_worker.clear_pending_approval")
    def test_rejection_clears_structured_flags(self, mock_clear, mock_get_model):
        """After rejection, waiting_for_approval=False and pending_file_approvals={}."""
        mock_decision = SupervisorDecision(
            plan=["Create file"], next_agent="synthesizer", current_task=""
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = self._base_hitl_state(user_message="No, cancel it")
        result = supervisor_node(state)

        assert result.get("waiting_for_approval") is False
        assert result.get("pending_file_approvals") == {}
        refs = result.get("scratchpad_references", [])
        assert any("[SYSTEM HITL]: User rejected" in r for r in refs)

    @patch("src.graph.supervisor.get_routing_model")
    def test_no_hitl_when_not_waiting(self, mock_get_model):
        """When waiting_for_approval=False, approval keywords in user message are ignored."""
        mock_decision = SupervisorDecision(
            plan=["Research"], next_agent="rag_worker", current_task="Find data"
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = self._base_hitl_state(user_message="approve", waiting=False, pending={})
        result = supervisor_node(state)

        # Should NOT contain any HITL notes since we weren't actually waiting
        refs = result.get("scratchpad_references", [])
        assert not any("SYSTEM HITL" in r for r in refs)

    @patch("src.graph.supervisor.get_routing_model")
    def test_waiting_without_decision_pauses_without_llm(self, mock_get_model):
        """When approval is pending but no decision exists, supervisor pauses instead of looping."""
        state = self._base_hitl_state(user_message="Create a helper file", waiting=True)
        result = supervisor_node(state)

        mock_get_model.assert_not_called()
        assert result["waiting_for_approval"] is True
        assert result["next_agent"] == "supervisor"
        assert result["pending_file_approvals"] == {"workspace/utils.py": {"tool": "create_files"}}
        assert any("Awaiting user approval" in ref for ref in result["scratchpad_references"])

    @patch("src.graph.supervisor.get_routing_model")
    @patch("src.agents.coding_worker.execute_pending_approval", return_value="Applied")
    @patch("src.agents.coding_worker.clear_pending_approval")
    @patch("src.tools.coding_tools._get_absolute_path", side_effect=lambda p: p)
    def test_approval_filepath_takes_precedence(self, mock_path, mock_clear, mock_exec, mock_get_model):
        """When approval_filepath is set, it should be used as the blocked file."""
        mock_decision = SupervisorDecision(
            plan=["Apply"], next_agent="synthesizer", current_task=""
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = self._base_hitl_state(user_message="yes")
        state["approval_filepath"] = "workspace/specific.py"
        state["pending_file_approvals"] = {"workspace/other.py": {"tool": "create_files"}}

        result = supervisor_node(state)

        # Should have approved workspace/specific.py (from approval_filepath), not workspace/other.py
        assert result.get("waiting_for_approval") is False


# ---------------------------------------------------------------------------
# 3. Loop Detection via worker_output_ids
# ---------------------------------------------------------------------------

class TestLoopDetection:
    """Verify supervisor evaluates worker_output_ids (not worker_outputs dict)
    for loop detection diagnostics."""

    @patch("src.graph.supervisor.get_routing_model")
    def test_retry_note_injected_when_output_ids_exist(self, mock_get_model):
        """When worker_output_ids is populated, a RETRY NOTE about cached outputs should be injected."""
        mock_decision = SupervisorDecision(
            plan=["Re-check data"], next_agent="rag_worker", current_task="Re-retrieve"
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="Find revenue data")],
            "plan": ["Find data"],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 8,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {"rag_worker": "wo_rag_abc123"},
            "worker_output_summaries": {"rag_worker": "Revenue is $10M"},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

        result = supervisor_node(state)

        # The model should have been invoked with a prompt containing the retry note
        call_args = mock_model.invoke.call_args[0][0]
        prompt_text = " ".join(
            msg.content for msg in call_args if hasattr(msg, "content")
        )
        assert "Existing worker outputs exist in memory" in prompt_text

    @patch("src.graph.supervisor.get_routing_model")
    def test_no_retry_note_when_output_ids_empty(self, mock_get_model):
        """When worker_output_ids is empty, no retry note about caching should appear."""
        mock_decision = SupervisorDecision(
            plan=["Find data"], next_agent="rag_worker", current_task="Research"
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="Find revenue data")],
            "plan": [],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 10,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

        result = supervisor_node(state)

        call_args = mock_model.invoke.call_args[0][0]
        prompt_text = " ".join(
            msg.content for msg in call_args if hasattr(msg, "content")
        )
        assert "Existing worker outputs exist in memory" not in prompt_text

    @patch("src.graph.supervisor.get_routing_model")
    def test_retry_counter_increments_on_failure(self, mock_get_model):
        """When retry_counter >= 1, it should increment and inject retry diagnostics."""
        mock_decision = SupervisorDecision(
            plan=["Retry task"], next_agent="web_worker", current_task="Try again"
        )
        mock_model = Mock()
        mock_model.with_config.return_value = mock_model
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="Search for data")],
            "plan": ["Search"],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 5,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 1,
            "context_notes": [],
        }

        result = supervisor_node(state)

        assert result["retry_counter"] == 2
        # Verify the retry note was injected
        call_args = mock_model.invoke.call_args[0][0]
        prompt_text = " ".join(
            msg.content for msg in call_args if hasattr(msg, "content")
        )
        assert "RETRY NOTE" in prompt_text


# ---------------------------------------------------------------------------
# 4. Multi-Hop Context Preservation
# ---------------------------------------------------------------------------

class TestMultiHopContextPreservation:
    """Verify scratchpad_references accumulates across multiple aggregation
    cycles, preserving multi-hop reasoning context."""

    def test_references_accumulate_across_runs(self):
        """Running aggregation twice should accumulate, not overwrite, references."""
        # First aggregation cycle
        state_1 = {
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "scratchpad_references": [],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {"rag_worker": "Revenue is $10M"},
            "scratchpad": "Step 1: Queried the RAG system",
        }
        result_1 = aggregate_parallel_results_node(state_1)

        assert len(result_1["scratchpad_references"]) >= 2  # scratchpad + rag_worker summary

        # Second aggregation cycle — feed previous references forward
        state_2 = {
            "worker_output_ids": result_1["worker_output_ids"],
            "worker_output_summaries": result_1["worker_output_summaries"],
            "scratchpad_references": result_1["scratchpad_references"],
            "active_document_ids": [],
            "task_hashes": [],
            "file_status_flags": {},
            "worker_outputs": {"web_worker": "Stock price is $45"},
            "scratchpad": "Step 2: Searched the web",
        }
        result_2 = aggregate_parallel_results_node(state_2)

        # All references from both cycles should be present
        all_refs = "\n".join(result_2["scratchpad_references"])
        assert "Step 1: Queried the RAG system" in all_refs
        assert "Step 2: Searched the web" in all_refs
        assert "Rag Worker" in all_refs
        assert "Web Worker" in all_refs

    def test_three_hop_accumulation(self):
        """Three sequential aggregation cycles should produce a rolling timeline."""
        refs = []
        ids = {}
        summaries = {}

        workers = [
            ("rag_worker", "First hop: found document"),
            ("web_worker", "Second hop: verified online"),
            ("critic_worker", "Third hop: cross-checked facts"),
        ]

        for worker_name, output in workers:
            state = {
                "worker_output_ids": ids,
                "worker_output_summaries": summaries,
                "scratchpad_references": refs,
                "active_document_ids": [],
                "task_hashes": [],
                "file_status_flags": {},
                "worker_outputs": {worker_name: output},
                "scratchpad": f"Processing {worker_name}",
            }
            result = aggregate_parallel_results_node(state)
            refs = result["scratchpad_references"]
            ids = result["worker_output_ids"]
            summaries = result["worker_output_summaries"]

        # All three hops should be present in the final references
        final_refs_text = "\n".join(refs)
        assert "First hop" in final_refs_text
        assert "Second hop" in final_refs_text
        assert "Third hop" in final_refs_text
        assert len(refs) >= 6  # 3 scratchpad entries + 3 worker summaries


# ---------------------------------------------------------------------------
# 5. Supervisor Graceful Empty State
# ---------------------------------------------------------------------------

class TestSupervisorEdgeCases:
    """Edge cases for supervisor robustness."""

    @patch("src.graph.supervisor.get_routing_model")
    def test_supervisor_handles_empty_state(self, mock_get_model):
        """Supervisor should not crash on a minimal/empty state."""
        mock_decision = SupervisorDecision(
            plan=["Start"], next_agent="rag_worker", current_task="Begin research"
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="Hello")],
            "plan": [],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 10,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "rag_worker"
        assert result["steps_remaining"] == 9
        assert isinstance(result["plan"], list)

    @patch("src.graph.supervisor.get_routing_model")
    def test_supervisor_clamps_steps_at_zero(self, mock_get_model):
        """Steps remaining should never go below 0."""
        mock_decision = SupervisorDecision(
            plan=["Finish"], next_agent="synthesizer", current_task=""
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="wrap up")],
            "plan": [],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 0,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

        result = supervisor_node(state)
        assert result["steps_remaining"] == 0  # max(0, 0-1) = 0

    @patch("src.graph.supervisor.get_routing_model")
    def test_supervisor_invalid_agent_defaults_to_synthesizer(self, mock_get_model):
        """When model returns an invalid agent name, should default to synthesizer."""
        mock_decision = SupervisorDecision(
            plan=["Finish"], next_agent="nonexistent_worker", current_task="invalid"
        )
        mock_model = Mock()
        mock_model.invoke.return_value = mock_decision
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="test")],
            "plan": [],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 5,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

        result = supervisor_node(state)
        assert result["next_agent"] == "synthesizer"

    @patch("src.graph.supervisor.get_routing_model")
    def test_supervisor_llm_failure_records_error(self, mock_get_model):
        """When the LLM call raises an exception, error should be recorded in scratchpad_references."""
        mock_model = Mock()
        mock_model.invoke.side_effect = Exception("429 rate_limit exceeded")
        mock_get_model.return_value = mock_model

        state = {
            "messages": [HumanMessage(content="test")],
            "plan": [],
            "scratchpad": "",
            "scratchpad_references": [],
            "steps_remaining": 5,
            "waiting_for_approval": False,
            "pending_file_approvals": {},
            "approval_filepath": "",
            "worker_output_ids": {},
            "worker_output_summaries": {},
            "file_status_flags": {},
            "task_hashes": [],
            "active_document_ids": [],
            "retry_counter": 0,
            "context_notes": [],
        }

        result = supervisor_node(state)

        # Error should be recorded in scratchpad_references
        refs = result.get("scratchpad_references", [])
        assert any("SYSTEM ERROR" in r for r in refs)
        # Should still default to synthesizer on failure
        assert result["next_agent"] == "synthesizer"
