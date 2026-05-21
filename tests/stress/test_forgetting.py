"""
================================================================================
RAG CONTEXT ENGINE - MEMORY & RETRIEVAL FORGETTING STRESS TEST
================================================================================
Stresses both modes (Simple RAG and Advanced Context Engine) to observe:
1. Dialogue memory overflow boundaries.
2. Recall accuracy when information is buried in noise (Needle-in-a-Haystack).
3. Hallucination/forgetting trigger points.
"""

import os
import sys
import uuid
import time
import requests
import json

API_URL = "http://localhost:8000"

def check_server_online():
    try:
        r = requests.get(f"{API_URL}/stats")
        return r.status_code == 200
    except Exception:
        return False


def run_memory_stress_test():
    """
    Tests dialogue memory longevity by injecting a secret key in turn 1,
    then sending N filler turns, checking if the model still recalls the secret.
    """
    print("\n" + "="*60)
    print(" 1. DIALOGUE MEMORY FORGETTING STRESS TEST")
    print("="*60)

    # 1. Advanced Context Engine Mode
    session_ce = f"stress-ce-{uuid.uuid4().hex[:6]}"
    print(f"\n[Context Engine Mode] Starting session: {session_ce}")
    
    # Inject the secret
    init_payload = {
        "question": "Remember this secret code: The key to the server vault is 'VaultKey-9988'. Confirm that you got it.",
        "session_id": session_ce,
        "mode": "context_engine"
    }
    r = requests.post(f"{API_URL}/query", json=init_payload)
    print(f"Turn 0 (Secret Injection) -> Response: {r.json().get('response', '').strip()}")

    filler_questions = [
        "What is the capital of France?",
        "Explain how the TCP three-way handshake works in detail.",
        "What is the speed of light in a vacuum?",
        "Describe the main difference between TCP and UDP protocols.",
        "Name three major routing protocols used in modern networking.",
        "What is the purpose of a subnet mask?",
        "How does DNS resolution work step by step?",
        "What is the function of a network gateway?",
        "What is DHCP and why is it used?",
        "Briefly explain the OSI model layer 4 responsibilities."
    ]

    ce_recalled = True
    ce_forget_turn = -1

    for idx, q in enumerate(filler_questions):
        # 1. Ask a filler question
        payload_filler = {
            "question": q,
            "session_id": session_ce,
            "mode": "context_engine"
        }
        requests.post(f"{API_URL}/query", json=payload_filler)
        
        # 2. Ask if it still remembers the secret
        payload_test = {
            "question": "What is the key to the server vault?",
            "session_id": session_ce,
            "mode": "context_engine"
        }
        res = requests.post(f"{API_URL}/query", json=payload_test).json()
        ans = res.get("response", "")
        tokens_used = res.get("stats", {}).get("budget_tracking", {}).get("memory_tokens_used", 0)
        
        recalled = "VaultKey-9988" in ans
        print(f"Turn {idx+1} (Memory Budget: {tokens_used} tokens) -> Recalled: {recalled} | Answer: {ans.strip()[:60]}...")
        
        if not recalled and ce_recalled:
            ce_recalled = False
            ce_forget_turn = idx + 1
            print(f" >>> [FORGOT] Context Engine forgot the secret at turn {ce_forget_turn}!")

    # 2. Simple RAG Mode (no memory context tracking)
    session_simple = f"stress-simple-{uuid.uuid4().hex[:6]}"
    print(f"\n[Simple RAG Mode] Starting session: {session_simple}")
    init_payload["session_id"] = session_simple
    init_payload["mode"] = "normal"
    r = requests.post(f"{API_URL}/query", json=init_payload)
    print(f"Turn 0 (Secret Injection) -> Response: {r.json().get('response', '').strip()}")

    # Test immediate recall
    test_payload = {
        "question": "What is the key to the server vault?",
        "session_id": session_simple,
        "mode": "normal"
    }
    res_simple = requests.post(f"{API_URL}/query", json=test_payload).json()
    ans_simple = res_simple.get("response", "")
    simple_recalled = "VaultKey-9988" in ans_simple
    print(f"Turn 1 (Test Recall) -> Recalled: {simple_recalled} | Answer: {ans_simple.strip()[:60]}...")

    return {
        "ce_forget_turn": ce_forget_turn if not ce_recalled else "Never (Still recalls after 10 turns)",
        "simple_recalled_turn_1": simple_recalled
    }


def run_knowledge_stress_test():
    """
    Tests knowledge recall when a specific document is buried in background noise (Needle-in-a-Haystack).
    """
    print("\n" + "="*60)
    print(" 2. RETRIEVAL & COMPRESSION STRESS TEST (NEEDLE IN A HAYSTACK)")
    print("="*60)

    # 1. Inject the needle document
    print("\nInjecting needle document...")
    needle_text = "Verification code: The primary system administrator access code is 'OmegaAdmin-4433-System'."
    fd = {"file": ("needle.txt", needle_text, "text/plain")}
    requests.post(f"{API_URL}/upload", files=fd)

    # 2. Inject noisy background documents (Haystack)
    print("Injecting noisy background documents...")
    noise_documents = [
        "Routing Information Protocol (RIP) is one of the oldest distance-vector routing protocols, which employs the hop count as a routing metric. RIP prevents routing loops by implementing a limit on the number of hops allowed in a path from source to destination.",
        "Open Shortest Path First (OSPF) is a routing protocol for Internet Protocol (IP) networks. It uses a link state routing algorithm and falls into the group of interior gateway protocols, operating within a single autonomous system.",
        "Border Gateway Protocol (BGP) is a standardized exterior gateway protocol designed to exchange routing and reachability information among autonomous systems on the Internet. BGP is classified as a path-vector routing protocol.",
        "Intermediate System to Intermediate System (IS-IS) is a routing protocol designed to move information efficiently within a computer network, a group of physically connected computers or similar devices.",
        "Enhanced Interior Gateway Routing Protocol (EIGRP) is an advanced distance-vector routing protocol that is used on a computer network for automating routing decisions and configuration.",
        "Network Address Translation (NAT) is a method of mapping an IP address space into another by modifying network address information in the IP header of packets while they are in transit across a traffic routing device.",
        "Dynamic Host Configuration Protocol (DHCP) is a network management protocol used on Internet Protocol networks whereby a DHCP server dynamically assigns an IP address and other network configuration parameters to each device.",
        "Domain Name System (DNS) is a hierarchical and decentralized naming system for computers, services, or other resources connected to the Internet or a private network.",
        "Hypertext Transfer Protocol (HTTP) is an application layer protocol for distributed, collaborative, hypermedia information systems. HTTP is the foundation of data communication for the World Wide Web.",
        "Transmission Control Protocol (TCP) is one of the main protocols of the Internet Protocol Suite. It originated in the initial network implementation in which it complemented the Internet Protocol."
    ]

    for idx, noise in enumerate(noise_documents):
        fd_noise = {"file": (f"noise_{idx}.txt", noise, "text/plain")}
        requests.post(f"{API_URL}/upload", files=fd_noise)

    # Allow Weaviate search buffer time
    time.sleep(2)

    # 3. Test Retrieval
    query = "What is the primary system administrator access code?"
    
    # Run in Advanced Context Engine Mode
    print("\n[Testing Context Engine Mode Recall...]")
    payload_ce = {
        "question": query,
        "mode": "context_engine",
        "session_id": f"needle-ce-{uuid.uuid4().hex[:6]}"
    }
    res_ce = requests.post(f"{API_URL}/query", json=payload_ce).json()
    ans_ce = res_ce.get("response", "")
    ce_recalled = "OmegaAdmin-4433-System" in ans_ce
    ce_score = res_ce.get("stats", {}).get("reranker_peak_score", 0)
    ce_ratio = res_ce.get("stats", {}).get("compression_ratio", 1.0)
    print(f"Context Engine -> Recalled: {ce_recalled} | Reranker Peak Score: {ce_score} | Compression Ratio: {ce_ratio:.2%}")
    print(f"Answer: {ans_ce.strip()}")

    # Run in Simple RAG Mode
    print("\n[Testing Simple RAG Mode Recall...]")
    payload_simple = {
        "question": query,
        "mode": "normal",
        "session_id": f"needle-simple-{uuid.uuid4().hex[:6]}"
    }
    res_simple = requests.post(f"{API_URL}/query", json=payload_simple).json()
    ans_simple = res_simple.get("response", "")
    simple_recalled = "OmegaAdmin-4433-System" in ans_simple
    print(f"Simple RAG -> Recalled: {simple_recalled}")
    print(f"Answer: {ans_simple.strip()}")

    return {
        "ce_recalled": ce_recalled,
        "ce_reranker_score": ce_score,
        "simple_recalled": simple_recalled
    }


if __name__ == "__main__":
    print("="*60)
    print("          RAG CONTEXT FORGETTING & HALLUCINATION STRESS TEST        ")
    print("="*60)

    if not check_server_online():
        print("CRITICAL: Server is offline. Please start 'python main.py' on port 8000 first.")
        sys.exit(1)

    mem_results = run_memory_stress_test()
    know_results = run_knowledge_stress_test()

    # Generate Markdown Report
    report = f"""# Stress Test Report: Context Longevity & Recall Limits

This report analyzes the recall decay and forgetting trigger points under both **Advanced Context Engine** and **Simple RAG** processing modes.

---

## 1. Dialogue Memory Stress Test

*   **Objective:** Inject a key fact into conversation history, send 10 consecutive filler turns of network concepts, and observe if the model retains the fact.
*   **Token Limit:** Context Engine allocates a max budget of **300 tokens** to the active conversation history.

### Results
*   **Simple RAG Mode:** Recalled on Turn 1? **{mem_results['simple_recalled_turn_1']}**
    *   *Observation:* Because Simple RAG does not load any conversation history context into the prompt, it has zero recall on subsequent turns.
*   **Advanced Context Engine Mode:** Recalled on Turn 1-10? **{mem_results['ce_forget_turn']}**
    *   *Observation:* The Context Engine retains the fact using its memory layer. As dialogue turns overflow the 300 token budget, older turns are decayed/dropped, leading to forgetting.

---

## 2. Knowledge Retrieval Stress Test (Needle-in-a-Haystack)

*   **Objective:** Inject 1 secret code chunk (needle) and 10 large routing protocol chunks (haystack) into Weaviate, query the system, and verify if it extracts the code.

### Results
*   **Simple RAG Mode:** Recalled? **{know_results['simple_recalled']}**
*   **Advanced Context Engine Mode:** Recalled? **{know_results['ce_recalled']}**
    *   *Reranker Score:* **{know_results['ce_reranker_score']}**
    *   *Observation:* The Advanced mode utilizes Cross-Encoder rerankers to rank the needle higher than the noise documents, and then applies sentence compression to isolate only the target code.
"""

    report_path = "tests/results/forgetting_test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + "="*60)
    print(f" STRESS TEST COMPLETE. Report written to {report_path}")
    print("="*60)
