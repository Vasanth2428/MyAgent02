import os
import time
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

from src.core.config import REPORT_WORKER_MODEL_PRIMARY, REPORT_WORKER_MODEL_FALLBACK

logger = logging.getLogger("MultiAgent.ReportWorker")

REPORT_SYSTEM_PROMPT = """You are a Report Specialist for a cooperative multi-agent system.
Your job is to take the accumulated findings (from the blackboard/scratchpad) and write a comprehensive, well-structured markdown report based on the user's specific request.

Formatting Guidelines:
1. Use markdown formatting (headers, lists, bold text) for clarity.
2. Provide a clear introduction, body paragraphs with synthesized points, and a conclusion.
3. Incorporate all relevant facts from the scratchpad.
4. Do NOT include any conversational filler like "Here is the report." Just output the raw markdown text of the report itself.
"""

def cleanup_old_reports(reports_dir: str, max_age_hours: int = 48):
    """Delete reports older than the specified max age to prevent disk bloat."""
    if not os.path.exists(reports_dir):
        return
        
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(reports_dir):
        filepath = os.path.join(reports_dir, filename)
        if os.path.isfile(filepath):
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    logger.info(f"Deleted old report: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to delete old report {filename}: {e}")

def get_report_model():
    """Get the LLM model for report generation."""
    primary_key = os.getenv("GROQ_CORE_KEY")
    api_key = primary_key or os.getenv("AGENT_API_KEY")
    primary = ChatGroq(model=REPORT_WORKER_MODEL_PRIMARY, temperature=0.3, api_key=api_key)
    fallback = ChatGroq(model=REPORT_WORKER_MODEL_FALLBACK, temperature=0.3, api_key=api_key)
    return primary.with_fallbacks([fallback])

def report_worker_node(state: dict) -> dict:
    """
    Report worker that generates a comprehensive markdown report and saves it locally.
    """
    logger.info("Report worker node executing...")
    
    current_task = state.get("current_task", "Generate a comprehensive report based on the findings.")
    scratchpad = state.get("scratchpad", "")
    worker_complete = state.get("worker_complete", {})
    
    model = get_report_model()
    
    prompt = f"""
Task Instruction: {current_task}

Accumulated Findings:
{scratchpad if scratchpad else "(No findings provided)"}

Please write the comprehensive report based on the above information.
"""
    
    try:
        response = model.invoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        report_content = response.content.strip()
        
        # Ensure reports directory exists and clean up old reports
        reports_dir = os.path.join(os.getcwd(), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        cleanup_old_reports(reports_dir)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.md"
        filepath = os.path.join(reports_dir, filename)
        
        # Save report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_content)
            
        summary_msg = f"Report successfully generated and saved to {filepath}."
        print(f"\n[REPORT WORKER] {summary_msg}")
        
    except Exception as e:
        logger.error(f"Error in report worker: {e}")
        summary_msg = f"Failed to generate report: {e}"
        report_content = ""

    updated_scratchpad = scratchpad + f"\n- [Report Worker]: {summary_msg}"
    worker_complete["report_worker"] = True

    return {
        "messages": [AIMessage(content=summary_msg, name="report_worker")],
        "scratchpad": updated_scratchpad,
        "worker_complete": worker_complete,
        "worker_outputs": {"report_worker": summary_msg},
        "worker_type": "report_worker",
        "next_agent": "supervisor"
    }
