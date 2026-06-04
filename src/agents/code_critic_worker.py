# Code critic worker node - validates symbol usage, checks for hallucinations, and audits patches.
import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

logger = logging.getLogger("MultiAgent.CodeCriticWorker")

CRITIC_SYSTEM_PROMPT = """You are a Code Critic and Security Auditor. Your job is to validate the findings and proposed code changes from the Coding Specialist worker.

You must examine:
1. Symbol references: Ensure no class, function, or method names mentioned by the worker are hallucinated. They must be validated against the actual repository symbols.
2. Code corrections / Diff patch correctness: Check if the patch diff aligns with the code structure.
3. Unsupported conclusions or logic flaws: Fact-check code reasoning and logic claims.

Check against the provided Repository Symbol List and Dependency details.
Output a structured analysis rating the severity of any found issues (Severity levels: 'info', 'warning', 'critical').
"""

class CriticFinding(BaseModel):
    issue_type: str = Field(description="The category of issue (e.g. 'hallucinated_symbol', 'syntax_error', 'unsupported_claim', 'patch_mismatch')")
    symbol_name: Optional[str] = Field(description="The symbol associated with the issue, if applicable", default=None)
    details: str = Field(description="Detailed explanation of the issue or validation finding")
    severity: str = Field(description="Severity rating ('info', 'warning', 'critical')")

class CriticReport(BaseModel):
    valid: bool = Field(description="True if the coding worker's findings and patches are fully validated with no critical issues.")
    findings: List[CriticFinding] = Field(description="Detailed checklist of individual validation findings")
    criticism_summary: str = Field(description="Overall review summary and critique of the work")


def get_critic_model():
    """Returns Groq model with structured output mapping for Critic reports."""
    model_name = os.getenv("REASONING_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("AGENT_API_KEY")
    llm = ChatGroq(model=model_name, temperature=0, api_key=api_key)
    return llm.with_structured_output(CriticReport)


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
    
    try:
        report: CriticReport = model.invoke(critic_prompt)
        
        # Format findings for presentation
        output_lines = []
        output_lines.append("### CODE CRITIC VALIDATION REPORT")
        output_lines.append(f"**Valid**: {report.valid}")
        output_lines.append(f"**Critique Summary**: {report.criticism_summary}\n")
        
        if report.findings:
            output_lines.append("### FINDINGS DETAIL")
            for f in report.findings:
                output_lines.append(f"- **[{f.severity.upper()}]** ({f.issue_type}): {f.details} (Symbol: {f.symbol_name or 'N/A'})")
        else:
            output_lines.append("- No issues detected.")
            
        final_text = "\n".join(output_lines)
    except Exception as e:
        logger.error(f"Error executing critic model call: {e}")
        final_text = f"Error during Code Critic model execution: {e}"

    logger.info("Code Critic Worker execution completed.")
    updated_scratchpad = scratchpad + f"\n- [Code Critic]: Code validation report:\n{final_text}"
    
    return {
        "messages": [AIMessage(content=final_text, name="code_critic_worker")],
        "scratchpad": updated_scratchpad,
        "worker_complete": {"code_critic_worker": True},
        "worker_outputs": {"code_critic_worker": final_text},
        "worker_type": "code_critic_worker",
        "next_agent": "supervisor"
    }
