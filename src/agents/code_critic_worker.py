# Code critic worker node - validates symbol usage, checks for hallucinations, and audits patches.
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from src.core.config import CODE_CRITIC_MODEL_PRIMARY, CODE_CRITIC_MODEL_FALLBACK

logger = logging.getLogger("MultiAgent.CodeCriticWorker")

CRITIC_SYSTEM_PROMPT = """You are a Code Critic and Security Auditor. Your job is to validate the findings and proposed code changes from the Coding Specialist worker.

You must examine:
1. Symbol references: Ensure no class, function, or method names mentioned by the worker are hallucinated. They must be validated against the actual repository symbols.
2. Code corrections / Diff patch correctness: Check if the patch diff aligns with the code structure and does not introduce security vulnerabilities.
3. Unsupported conclusions or logic flaws: Fact-check code reasoning and logic claims.
4. Security audit: Review for injection vulnerabilities, hardcoded secrets, unsafe patterns, and privilege escalation risks.

Check against the provided Repository Symbol List and Dependency details.
Output a structured analysis rating the severity of any found issues (Severity levels: 'info', 'warning', 'critical').

If you detect a critical issue that MUST be fixed, end your response with the exact token 'RETRY_REQUIRED'.
"""

class CriticFinding(BaseModel):
    issue_type: str = Field(description="The category of issue (e.g. 'hallucinated_symbol', 'syntax_error', 'unsupported_claim', 'patch_mismatch', 'security_risk')")
    symbol_name: Optional[str] = Field(description="The symbol associated with the issue, if applicable", default=None)
    file_location: Optional[str] = Field(description="The file location where the issue was found", default=None)
    details: str = Field(description="Detailed explanation of the issue or validation finding")
    severity: str = Field(description="Severity rating ('info', 'warning', 'critical')")
    evidence: Optional[str] = Field(description="Evidence supporting the finding", default=None)

class CriticReport(BaseModel):
    valid: bool = Field(description="True if the coding worker's findings and patches are fully validated with no critical issues.")
    findings: List[CriticFinding] = Field(description="Detailed checklist of individual validation findings")
    criticism_summary: str = Field(description="Overall review summary and critique of the work")


def get_critic_model():
    """Returns Groq model with structured output mapping for Critic reports."""
    validation_key = os.getenv("GROQ_VALIDATION_KEY")
    api_key = validation_key or os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=CODE_CRITIC_MODEL_PRIMARY, temperature=0, api_key=api_key).with_structured_output(CriticReport)
    fallback = ChatGroq(model=CODE_CRITIC_MODEL_FALLBACK, temperature=0, api_key=api_key).with_structured_output(CriticReport)
    return primary.with_fallbacks([fallback])


def code_critic_worker_node(state: dict) -> dict:
    """
    Code critic worker that audits coding worker outputs against repository symbol tables.
    """
    logger.info("Executing Code Critic Worker node...")
    
    # 1. Access current blackboard findings
    scratchpad = state.get("scratchpad", "")
    current_task = state.get("current_task", "")
    worker_outputs = state.get("worker_outputs", {})
    
    # Get coding worker's output
    coding_output = worker_outputs.get("coding_worker", "")
    if not coding_output:
        logger.warning("No coding worker output detected to critique. Skipping validation.")
        return {
            "worker_complete": {"code_critic_worker": True},
            "worker_outputs": {"code_critic_worker": "No coding specialist output was found to validate."},
            "worker_type": "code_critic_worker",
            "next_agent": "supervisor"
        }
        
    # 2. Extract repository index symbols for validation context
    from src.agents.coding_worker import get_retrieval_service
    try:
        service = get_retrieval_service()
        all_symbols = service.indexer.symbol_table.get_all_symbols()
        symbol_details = [
            f"[{s['type'].upper()}] {s['name']} in {s['filepath']}:{s['start_line']}-{s['end_line']}"
            for s in all_symbols
        ]
        repo_context = "Repository Available Symbols:\n" + "\n".join(symbol_details)
    except Exception as e:
        logger.error(f"Failed to load repository symbols for critic: {e}")
        repo_context = "Repository Available Symbols: (Failed to load symbols)"

    # 3. Invoke structured LLM review
    model = get_critic_model()
    
    critic_prompt = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        SystemMessage(content=repo_context),
        HumanMessage(content=f"Coding Specialist Task: {current_task}\n\nCoding Specialist Output:\n{coding_output}")
    ]
    
    is_invalid = False
    try:
        report: CriticReport = model.invoke(critic_prompt)
        
        # Format findings for presentation
        output_lines = []
        output_lines.append("### CODE CRITIC VALIDATION REPORT")
        output_lines.append(f"**Status**: {'✓ VALIDATED' if report.valid else '✗ ISSUES FOUND'}")
        output_lines.append(f"**Critique Summary**: {report.criticism_summary}\n")
        
        if report.findings:
            output_lines.append("### FINDINGS DETAIL")
            output_lines.append("")
            for f in report.findings:
                loc = f" ({f.file_location})" if f.file_location else ""
                output_lines.append(f"- **[{f.severity.upper()}]** ({f.issue_type}){loc}")
                output_lines.append(f"  - {f.details}")
                if f.evidence:
                    output_lines.append(f"  - Evidence: {f.evidence}")
                if f.symbol_name:
                    output_lines.append(f"  - Symbol: {f.symbol_name}")
                output_lines.append("")
        else:
            output_lines.append("No issues detected.")
        
        is_invalid = not report.valid or any(f.severity.lower() == "critical" for f in report.findings)
        retry_count = state.get("critic_retry_count", 0)
        
        if is_invalid:
            if retry_count < 2:
                output_lines.append("\nRETRY_REQUIRED")
            else:
                output_lines.append("\n[Max validation retry limit reached. Verification failed after multiple attempts. Proceeding without further retries.]")
             
        final_text = "\n".join(output_lines)
    except Exception as e:
        logger.error(f"Error executing critic model call: {e}")
        final_text = f"Error during Code Critic model execution: {e}"
        is_invalid = False
        retry_count = state.get("critic_retry_count", 0)

    logger.info("Code Critic Worker execution completed.")
    
    if is_invalid and retry_count >= 2:
        updated_scratchpad = scratchpad + f"\n- [Code Critic]: Verification failed repeatedly. Aborting corrections to prevent infinite loop.\nFindings:\n{final_text}"
    else:
        updated_scratchpad = scratchpad + f"\n- [Code Critic]: Code validation report:\n{final_text}"
    
    state_update = {
        "messages": [AIMessage(content=final_text, name="code_critic_worker")],
        "scratchpad": updated_scratchpad,
        "worker_complete": {"code_critic_worker": True},
        "worker_outputs": {"code_critic_worker": final_text},
        "worker_type": "code_critic_worker",
        "next_agent": "supervisor",
        "critic_retry_count": retry_count
    }
    
    if is_invalid and retry_count < 2:
        logger.info(f"[CODE CRITIC WORKER] Critical issue detected! Forcing supervisor retry (retry {retry_count + 1}/2).")
        current_plan = state.get("plan", [])
        state_update["plan"] = current_plan + ["FIX ERROR: Review code critic feedback and modify files to correct the issues."]
        state_update["critic_retry_count"] = retry_count + 1
        
    return state_update
