# Real End-to-End System Test

*Generated on: 2026-06-06T03:08:03.417093*

**Overall: ALL PASS**

| Check | Result | Detail |
|---|---|---|
| Server responding | PASS |  |
| Engine online | PASS | status=online |
| Documents indexed | PASS | count=898 |
| CPU readable | PASS | cpu=10.0% |
| Request succeeded | PASS | status=200 |
| Got a non-empty answer | PASS | len=649 chars |
| Answer mentions core concepts | PASS | Answer: The 1% rule in Atomic Habits refers to the idea that small, incremental  |
| LLM used context (not hallucinating) | PASS | retrieved_context present: True |
| Latency acceptable (<20s) | PASS | latency=3.98s |
| Query count incremented | PASS | queries=1 |
| First IKIGAI query succeeded | PASS |  |
| Answer mentions Ikigai concept | PASS | snippet: Unfortunately, the provided context does not explicitly mention the fou |
| Follow-up query succeeded | PASS |  |
| Follow-up answer is non-empty | PASS | len=392 |
| Follow-up references context | PASS | snippet: To discover what feels natural to you, ignore what you have been taught |
| History persisted in DB | PASS | history entries=4 |
| SQL aggregation returns data | PASS | result snippet: | status | count | total |
|--------|-------|-------|
| Complete |
| Customer count query works | PASS | result: | total |
|-------|
| 8 | |
| DROP blocked | PASS | response: Error: Only SELECT queries are permitted. The keyword 'DROP' is forbid |
| UNION blocked | PASS | response: Error: Only SELECT queries are permitted. The keyword 'UNION' is forbi |
| Web search returned results | PASS | count=3 |
| Results have real URLs (not mock) | PASS | urls=['https://www.ibm.com/think/topics/retrieval-augmented-generat', 'https://w |
| Results have real content | PASS | content lengths=[1375, 1022, 139] |
| Latency under 10s | PASS | latency=1.70s |
| Create session | PASS | status=201 |
| Session ID returned | PASS | sid=SID-6BFDBDCB |
| List sessions works | PASS |  |
| New session in list | PASS | session count=7 |
| Rename session | PASS |  |
| Rename persisted in DB | PASS | title=Renamed E2E Session |
| Delete session | PASS |  |
| Session gone after delete | PASS |  |
| Stream connected | PASS | status=200 |
| Received streaming tokens | PASS | chunk_count=50 |
| Assembled answer non-empty | PASS | len=256 |
| Stream finished with done event | PASS |  |
| Stream latency ok (<25s) | PASS | latency=2.92s |
