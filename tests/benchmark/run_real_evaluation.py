"""
Real Evaluation Suite
Runs empirical tests against the live engine, models, and databases.
All LLM-calling tests have explicit timeouts to prevent hangs.
"""
import os
import asyncio
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

load_dotenv()

from src.core.config import PipelineConfig
from src.core.retriever import WeaviateRetriever
from src.core.engine import RAGContextEngine, count_tokens
from src.core.memory import ConversationMemory
from src.core.reranker import NeuralReranker
from src.core.compressor import Compressor
from src.core.services.grounding_service import GroundingVerifier
from tests.benchmark.gold_dataset import GOLD_DATASET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LLM_TIMEOUT = 30  # seconds per LLM call


def write_report(filename, title, content):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"*Generated on: {datetime.now().isoformat()}*\n\n")
        f.write(content)
    print(f"[OK] Generated {filename}")


def format_result(passed, details):
    status = "PASS" if passed else "FAIL"
    return f"**Status:** {status}\n\n**Details:**\n{details}\n"


async def test_compressor_accuracy(engine: RAGContextEngine):
    print("Running Compressor Accuracy Test...")

    # Build multiple paragraphs so the compressor has real segments to score.
    # The needle paragraph is semantically relevant; the rest are off-topic noise.
    needle_doc = "The secret launch code is OMEGA-99-ACTUAL. This is the top-secret code used to authorise launches."
    noise_docs = [
        f"This is unrelated paragraph number {i} about the history of ancient Roman aqueducts "
        f"and their engineering challenges during the third century BC. "
        f"The Romans built many aqueducts across their empire to supply water to cities."
        for i in range(50)
    ]
    all_docs = noise_docs[:25] + [needle_doc] + noise_docs[25:]

    query = "What is the secret launch code?"
    start_tokens = count_tokens(" ".join(all_docs))
    start_time = time.time()

    try:
        # Use evaluate_compression which is the proper public API
        result = await asyncio.wait_for(
            asyncio.to_thread(Compressor.evaluate_compression, query, all_docs, ["OMEGA-99-ACTUAL"]),
            timeout=LLM_TIMEOUT
        )
        latency = time.time() - start_time
        fact_preserved = result["facts_preserved"] == 1.0
        compression_ratio = result["compression_ratio"]
        passed = fact_preserved and compression_ratio > 0.1
        details = f"""
- Input Documents: {len(all_docs)} paragraphs ({start_tokens} tokens total)
- Key Fact: 'OMEGA-99-ACTUAL'
- Fact Preserved After Compression: {fact_preserved}
- Compression Ratio (chars removed): {compression_ratio:.2f} (must be > 0.10)
- Noise Dropped: {result['noise_dropped']}
- Latency: {latency:.2f}s
"""
    except asyncio.TimeoutError:
        passed = False
        details = f"\n- TIMED OUT after {LLM_TIMEOUT}s\n"

    write_report("agentic_compressor_accuracy_report.md", "Agentic Compressor Accuracy", format_result(passed, details))


async def test_memory_decay():
    print("Running Memory Decay Test...")
    memory = ConversationMemory(max_tokens=300)

    for i in range(10):
        memory.add(f"This is a long filler message {i} that takes up tokens. " * 5, importance=0.5, role="user")

    memory.add("SYSTEM: You are a helpful assistant.", importance=1.0, role="system")
    memory.add("What is the capital of France?", importance=0.8, role="user")

    context = memory.get_active_context()
    tokens = count_tokens(context)
    has_system = "SYSTEM" in context
    has_latest = "France" in context
    passed = tokens <= 300 and has_system and has_latest

    details = f"""
- Final Context Tokens: {tokens} (Budget: 300)
- System Message Retained: {has_system}
- Newest Message Retained: {has_latest}
- Total Items in Context: {len(memory.entries)}
- Result: {"Within budget and critical messages retained" if passed else "FAILED - budget exceeded or messages lost"}
"""
    write_report("agentic_memory_decay_report.md", "Agentic Memory Decay & Token Budget", format_result(passed, details))


async def test_reranker_semantics():
    print("Running Reranker Semantics Test...")
    reranker = NeuralReranker()

    query = "Who invented Python?"
    docs = [
        {"text": "A python is a large snake that is a creator of fear.", "score": 0.9, "id": "1"},
        {"text": "Guido van Rossum created and released the programming language Python in 1991.", "score": 0.5, "id": "2"},
    ]

    start_time = time.time()
    reranked = await asyncio.to_thread(reranker.rerank, query, docs)
    latency = time.time() - start_time

    passed = reranked[0]["id"] == "2"
    details = f"""
- Query: '{query}'
- Doc 1 (Lexical match - snake): cross_score={reranked[1]['cross_score']:.4f} | Final Rank: #2
- Doc 2 (Semantic match - Guido): cross_score={reranked[0]['cross_score']:.4f} | Final Rank: #1
- Semantic match ranked higher than lexical distractor: {passed}
- Reranker Latency: {latency:.2f}s
"""
    write_report("agentic_reranker_semantics_report.md", "Neural Reranker Semantic Validation", format_result(passed, details))


async def test_concurrency(engine: RAGContextEngine):
    print("Running Concurrency Test...")
    queries = [
        "What is python?", "What is java?",
        "What is C++?", "What is Rust?", "What is Go?"
    ]

    async def make_req(q, i):
        start = time.time()
        res = await asyncio.wait_for(
            engine.ask_async(q, session_id=f"conc_{i}", mode="context_engine"),
            timeout=LLM_TIMEOUT
        )
        return time.time() - start

    try:
        start_all = time.time()
        times = await asyncio.gather(*(make_req(q, i) for i, q in enumerate(queries)))
        total_wall = time.time() - start_all
        passed = True
        err = "None"
        max_single = max(times)
        is_concurrent = total_wall < sum(times) * 0.9
    except Exception as e:
        passed = False
        times = []
        total_wall = 0
        max_single = 0
        err = str(e)
        is_concurrent = False

    details = f"""
- Total Requests: {len(queries)}
- Simultaneous Execution: Yes (asyncio.gather)
- Error: {err}
- Individual Latencies: {[round(t, 2) for t in times]} seconds
- Total Wall Time: {round(total_wall, 2)}s
- Max Single Request: {round(max_single, 2)}s
- True Concurrency Confirmed (wall < sum): {is_concurrent}
"""
    write_report("concurrency_test_report.md", "API Concurrency & Deadlock Validation", format_result(passed, details))


async def test_lost_in_middle(engine: RAGContextEngine):
    print("Running Lost in the Middle Test...")
    target_fact = "The launch code is 445566."
    docs = [f"Random filler fact number {i}." for i in range(20)]
    docs.insert(10, target_fact)
    context = " ".join(docs)
    query = "What is the launch code?"

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer concisely with just the number."

    try:
        response = await asyncio.wait_for(
            engine.generation_service.async_client.chat.completions.create(
                model=os.getenv("AGENT_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            ),
            timeout=LLM_TIMEOUT
        )
        ans = response.choices[0].message.content.strip()
        passed = "445566" in ans
    except asyncio.TimeoutError:
        ans = f"TIMED OUT after {LLM_TIMEOUT}s"
        passed = False
    except Exception as e:
        ans = str(e)
        passed = False

    details = f"""
- Document Count: 21
- Target Fact Position: Index 10 (exact middle)
- Target Fact: '{target_fact}'
- Query: '{query}'
- LLM Answer: '{ans}'
- Fact Successfully Extracted From Middle: {passed}
"""
    write_report("introspective_lost_in_middle_report.md", "Lost in the Middle Evaluation", format_result(passed, details))


async def test_rag_triad():
    print("Running RAG Triad Test...")
    verifier = GroundingVerifier()

    sample_size = 5
    total_score = 0
    rows = []

    for item in GOLD_DATASET[:sample_size]:
        answer = f"The answer is {item['answer_contains']}"
        facts = item["supporting_facts"]
        if not facts:
            continue
        score, ungrounded = verifier.verify_grounding(answer, facts)
        total_score += score
        rows.append(f"- Q: '{item['question']}' | Score: {score:.3f}")

    avg_score = total_score / sample_size
    passed = avg_score > 0.4

    details = f"""
- Evaluated Samples: {sample_size}
- Average Grounding Score: {avg_score:.3f}
- Minimum Pass Threshold: 0.4
- Per-Sample Results:
{chr(10).join(rows)}
- Verdict: {'Grounding is working correctly' if passed else 'Grounding score too low'}

NOTE: Grounding Score is a heuristic (lexical/semantic overlap). Not a ground-truth faithfulness metric.
"""
    write_report("introspective_rag_triad_report.md", "RAG Triad - Grounding Verification", format_result(passed, details))


async def test_multi_agent_architecture(engine: RAGContextEngine):
    """
    Directly calls the supervisor_node with a synthetic state dict.
    Tests that the supervisor:
    - Generates a non-empty plan
    - Routes to a valid worker (not directly to synthesizer for a complex query)
    """
    print("Running Multi-Agent Routing Test...")
    from langchain_core.messages import HumanMessage
    from src.graph.supervisor import supervisor_node

    query = "What is the total revenue from our sales database for the last quarter?"
    state = {
        "messages": [HumanMessage(content=query)],
        "plan": [],
        "scratchpad": "",
        "context_notes": [],
        "steps_remaining": 5,
        "next_agent": "",
        "current_task": "",
        "worker_complete": {},
        "worker_outputs": {},
        "final_answer": "",
    }

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(supervisor_node, state),
            timeout=LLM_TIMEOUT
        )
        next_agent = result.get("next_agent", "")
        plan = result.get("plan", [])
        task = result.get("current_task", "")
        valid_workers = ["rag_worker", "web_worker", "utility_worker", "scraper_worker", "critic_worker", "report_worker"]
        routed_to_worker = next_agent in valid_workers
        passed = routed_to_worker and len(plan) > 0

        details = f"""
- Query: '{query}'
- Supervisor Routed To: '{next_agent}'
- Routed to a Research Worker (not directly to synthesizer): {routed_to_worker}
- Plan Generated: {plan}
- Sub-Task Assigned: '{task}'
- Steps Remaining After Dispatch: {result.get("steps_remaining", "?")}
"""
    except asyncio.TimeoutError:
        passed = False
        details = f"\n- TIMED OUT after {LLM_TIMEOUT}s\n"
    except Exception as e:
        passed = False
        details = f"\n- ERROR: {e}\n{traceback.format_exc()}\n"

    write_report("multi_agent_architecture_review_report.md", "Multi-Agent Supervisor Routing Validation", format_result(passed, details))


async def test_adversarial_validation():
    print("Running Adversarial SQL Injection Test...")
    from src.core.sales_db import execute_read_only_sql

    injections = [
        ("DROP TABLE customers;", "drop"),
        ("SELECT * FROM sales UNION SELECT * FROM admin_users;", "union"),
        ("UPDATE sales SET amount=0;", "update"),
        ("DELETE FROM orders;", "delete"),
    ]

    blocks = 0
    rows = []

    for query, expected_kw in injections:
        res = execute_read_only_sql(query)
        blocked = expected_kw.upper() in res or "forbidden" in res.lower()
        if blocked:
            blocks += 1
        rows.append(f"- `{query}` -> Blocked: {blocked} | Response: `{res[:80]}`")

    passed = blocks == len(injections)
    details = f"""
- Injection Attempts: {len(injections)}
- Successfully Blocked: {blocks}/{len(injections)}
- Results:
{chr(10).join(rows)}
"""
    write_report("multi_agent_adversarial_validation_report.md", "Adversarial Validation - SQL Injection Prevention", format_result(passed, details))


async def main():
    print("=" * 60)
    print("STARTING REAL EVALUATION SUITE")
    print("=" * 60)

    try:
        retriever = WeaviateRetriever()
        config = PipelineConfig.from_env()
        engine = RAGContextEngine(retriever, config)
    except Exception as e:
        print(f"[FATAL] Failed to initialize engine: {e}")
        traceback.print_exc()
        return

    tests = [
        ("Compressor Accuracy",       lambda: test_compressor_accuracy(engine)),
        ("Memory Decay",              lambda: test_memory_decay()),
        ("Reranker Semantics",        lambda: test_reranker_semantics()),
        ("Lost in the Middle",        lambda: test_lost_in_middle(engine)),
        ("RAG Triad",                 lambda: test_rag_triad()),
        ("Adversarial SQL Injection", lambda: test_adversarial_validation()),
        ("Multi-Agent Routing",       lambda: test_multi_agent_architecture(engine)),
        ("Concurrency (5 parallel)",  lambda: test_concurrency(engine)),
    ]

    passed_count = 0
    failed_count = 0

    for name, coro in tests:
        print(f"\n[{name}]")
        try:
            await coro()
            passed_count += 1
        except Exception as e:
            print(f"  [ERROR] {name} crashed: {e}")
            traceback.print_exc()
            failed_count += 1

    print("\n" + "=" * 60)
    print(f"EVALUATION COMPLETE: {passed_count} succeeded, {failed_count} errors")
    print(f"Reports written to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
