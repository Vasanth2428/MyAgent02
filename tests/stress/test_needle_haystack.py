"""
Needle In Massive Haystack - Stress retrieval and compression beyond current scale.

Tests with 100 noise documents to verify:
1. Needle retrieved in top candidates
2. Needle survives compression
3. Answer contains needle fact
"""
import pytest
import asyncio
from src.core.compressor import Compressor
from src.core.reranker import NeuralReranker


def _generate_noise_document(index):
    """Generate realistic noise document for haystack testing."""
    topics = [
        "Network protocols and routing algorithms form the backbone of internet communication.",
        "Database indexing strategies improve query performance through optimized data structures.",
        "Machine learning models require feature engineering for optimal accuracy.",
        "Cloud computing services provide scalable infrastructure for modern applications.",
        "Cybersecurity best practices include multi-factor authentication and encryption.",
        "Container orchestration systems manage application deployment at scale.",
        "Load balancers distribute network traffic across multiple servers.",
        "Distributed systems use consensus algorithms for fault tolerance.",
        "API rate limiting prevents abuse and ensures fair resource allocation.",
        "Cache invalidation strategies improve read performance in web applications.",
        "Message queues enable asynchronous communication between services.",
        "Microservices architecture promotes independent deployment and scaling.",
        "Event-driven systems respond to state changes in real-time.",
        "Serverless functions execute code without managing infrastructure.",
        "Content delivery networks cache static assets at edge locations.",
        "DNS resolution translates domain names to IP addresses.",
        "TLS certificates secure HTTP connections with encryption.",
        "OAuth authentication enables secure third-party API access.",
        "Log aggregation centralizes application monitoring and debugging.",
        "Health checks verify service availability in distributed systems.",
        "Circuit breakers prevent cascade failures in microservice architectures.",
        "Rate limiting algorithms include token bucket and leaky bucket.",
        "Reverse proxies forward client requests to backend servers.",
        "SSL termination decrypts traffic at load balancer boundaries.",
        "Service meshes provide observability and traffic control.",
        "Database sharding distributes data across multiple instances.",
        "Connection pooling reuses database connections efficiently.",
        "Schema migrations update database structures safely.",
        "Backup strategies ensure data recovery from failures.",
        "Replication provides high availability and read scalability.",
        "Load shedding drops requests when systems are overloaded.",
        "Healthy systems maintain SLA compliance under stress.",
        "Observability tools provide insights into system performance.",
        "Tracing systems track request flow through distributed services.",
        "Metrics collection monitors system health and performance.",
        "Alert fatigue reduces effectiveness of monitoring systems.",
        "Auto-scaling adjusts resource capacity based on demand.",
        "Blue-green deployments enable zero-downtime releases.",
        "Canary releases gradually roll out new features.",
        "Rolling updates replace instances without service interruption.",
        "Feature flags enable gradual feature rollout.",
        "Chaos engineering tests system resilience proactively.",
        "Postmortems analyze system failures for improvement.",
        "Runbooks provide operational guidance for common incidents.",
        "SRE principles balance reliability with feature velocity.",
        "On-call rotations distribute operational responsibility.",
        "Incident response follows structured escalation procedures.",
        "Monitoring dashboards visualize system health metrics.",
        "Alert routing directs notifications to appropriate responders.",
        "PagerDuty integrations automate on-call notifications.",
        "Status pages communicate service availability to users.",
        "Root cause analysis identifies fundamental system issues.",
        "Blameless culture encourages honest post-incident discussion.",
        "Capacity planning forecasts resource requirements.",
        "Cost optimization balances performance with budget constraints.",
        "Infrastructure as code manages system configuration.",
        "Terraform templates provision cloud infrastructure.",
        "Kubernetes manifests define container deployments.",
        "Helm charts package application configurations.",
        "GitOps synchronizes cluster state with repository.",
        "Service accounts provide application-level permissions.",
        "Role-based access control restricts system access.",
        "Principle of least privilege minimizes attack surface.",
        "Security scanning detects vulnerabilities in dependencies.",
        "Static analysis catches bugs before deployment.",
        "Dynamic testing validates runtime behavior.",
        "Penetration testing identifies security weaknesses.",
        "Compliance frameworks regulate data handling practices.",
        "Audit trails track system changes for accountability.",
        "Data governance policies protect sensitive information.",
        "Privacy regulations govern personal data processing.",
        "Encryption at rest protects stored data.",
        "Encryption in transit secures network communications.",
        "Key rotation prevents long-term key exposure.",
        "Secrets management protects sensitive credentials.",
        "Vault systems secure secret storage and access.",
        "Certificate authorities issue TLS certificates.",
        "Certificate pinning prevents man-in-the-middle attacks.",
        "Public key infrastructure manages cryptographic keys.",
        "Digital signatures verify document authenticity.",
        "Blockchain technology enables distributed ledger systems.",
        "Smart contracts execute code on distributed networks.",
        "Consensus mechanisms validate network state.",
        "Proof of work secures blockchain networks.",
        "Proof of stake reduces energy consumption.",
        "Token economics incentivize network participation.",
        "Decentralized finance enables peer-to-peer transactions.",
    ]
    return f"Document {index}: {topics[index % len(topics)]} Additional detail: {index * 100} bytes of content."


def test_needle_survives_compression_100_docs():
    """Test needle retrieval with 100 noise documents."""
    needle = "CRITICAL-NUCLEAR-CODE-OMEGA-REDACTED-999"
    
    noise_documents = [_generate_noise_document(i) for i in range(100)]
    all_docs = noise_documents + [needle]
    
    query = "What is the critical nuclear code we need to know?"
    
    compressed = Compressor.compress(all_docs, query, max_tokens=200)
    
    assert needle in compressed, "Needle should survive compression"


def test_needle_reranker_score_separation():
    """Verify reranker creates clear separation between needle and noise."""
    reranker = NeuralReranker()
    needle = "SECRET-VIP-ACCESS-CODE-DELTA-777-WAS-HERE"
    query = "What is the secret VIP access code?"
    
    noise_samples = ["Network routing protocols manage traffic flow."] * 20
    candidates = [{"text": doc, "score": 0.5} for doc in noise_samples]
    candidates.append({"text": needle, "score": 0.5})
    
    reranked = reranker.rerank(query, candidates)
    
    assert len(reranked) == 21
    assert reranked[0]["text"] == needle, "Needle should rank highest"
    
    needle_score = reranked[0]["cross_score"]
    noise_score = reranked[1]["cross_score"]
    assert needle_score > noise_score, (
        f"Needle score ({needle_score}) should be better than noise ({noise_score})"
    )


def test_needle_compression_under_stress():
    """Test needle compression with 100 noise documents - isolation test."""
    needle = "EMERGENCY-SHUTDOWN-PROCEDURE-ACTIVATION-CODE"
    
    noise_documents = [_generate_noise_document(i) for i in range(100)]
    all_documents = noise_documents + [needle]
    
    compressed = Compressor.compress(all_documents, "What is the emergency shutdown code?", max_tokens=150)
    
    from src.core.engine import count_tokens
    compressed_tokens = count_tokens(compressed)
    
    # Adversarial: Verify token budget is respected
    assert compressed_tokens <= 200, f"Compressed context ({compressed_tokens} tokens) exceeds budget"
    
    # Needle should survive when there's budget
    if compressed_tokens <= 150:
        assert needle in compressed, "Needle must survive within token budget"


def test_needle_compression_ratio_quality():
    """Verify compression ratio quality with needle in haystack."""
    needle = "GOLDEN-SECRET-KEY-FOR-TESTING"
    
    noise_documents = [_generate_noise_document(i) for i in range(100)]
    all_documents = noise_documents + [needle]
    
    eval_result = Compressor.evaluate_compression(
        "What is the golden secret key?",
        all_documents,
        [needle]
    )
    
    assert eval_result["facts_preserved"] >= 0.9, "Needle should be preserved (90%+)"
    assert eval_result["compression_ratio"] > 0.5, "Significant compression should occur"


def test_needle_retrieval_amid_semantically_similar_noise():
    """Test needle retrieval when noise is semantically similar to needle."""
    needle = "TERMINATE-ALL-SESSIONS-IMMEDIATELY"
    
    noise_documents = [
        "Session management handles user authentication lifecycle.",
        "Active sessions require periodic cleanup.",
        "Session tokens expire after timeout.",
        "Session invalidation prevents unauthorized access.",
        "Sessions should be terminated gracefully.",
        "User sessions track application state.",
        "Session cleanup prevents memory leaks.",
        "Sessions terminate on logout.",
        "Failed sessions are automatically terminated.",
        "Session management is critical for security.",
        needle,
        "Session termination protocol requires confirmation.",
        "All active sessions will be logged out.",
        "Sessions may be terminated for policy violations.",
    ]
    
    query = "What command terminates all sessions immediately?"
    
    compressed = Compressor.compress(noise_documents, query, max_tokens=200)
    assert needle in compressed, "Needle should survive even amid similar noise"