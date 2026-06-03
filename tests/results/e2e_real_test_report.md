# Real End-to-End System Test

*Generated on: 2026-06-03T20:21:57.057295*

**Overall: ALL PASS**

| Check | Result | Detail |
|---|---|---|
| Server responding | PASS |  |
| Engine online | PASS | status=online |
| Documents indexed | PASS | count=839 |
| CPU readable | PASS | cpu=5.6% |
| Request succeeded | PASS | status=200 |
| Got a non-empty answer | PASS | len=532 chars |
| Answer mentions core concepts | PASS | Answer: The 1% rule in Atomic Habits refers to making a tiny change, a 1 percent |
| LLM used context (not hallucinating) | PASS | retrieved_context present: True |
| Latency acceptable (<20s) | PASS | latency=4.17s |
| Query count incremented | PASS | queries=8 |
| First IKIGAI query succeeded | PASS |  |
| Answer mentions Ikigai concept | PASS | snippet: The Japanese concept of Ikigai is a philosophy that roughly translates  |
| Follow-up query succeeded | PASS |  |
| Follow-up answer is non-empty | PASS | len=392 |
| Follow-up references context | PASS | snippet: To discover what feels natural to you, ignore what you have been taught |
| History persisted in DB | PASS | history entries=8 |
| SQL aggregation returns data | PASS | result snippet: | status | count | total |
|--------|-------|-------|
| Complete |
| Customer count query works | PASS | result: | total |
|-------|
| 8 | |
| DROP blocked | PASS | response: Error: Only SELECT queries are permitted. The keyword 'DROP' is forbid |
| UNION blocked | PASS | response: Error: Only SELECT queries are permitted. The keyword 'UNION' is forbi |
| Web search returned results | PASS | count=3 |
| Results have real URLs (not mock) | PASS | urls=['https://www.youtube.com/watch?v=qN_2fnOPY-M', 'https://weaviate.io/blog/i |
| Results have real content | PASS | content lengths=[3126, 1064, 1375] |
| Latency under 10s | PASS | latency=2.23s |
| Create session | PASS | status=201 |
| Session ID returned | PASS | sid=SID-084B238E |
| List sessions works | PASS |  |
| New session in list | PASS | session count=5 |
| Rename session | PASS |  |
| Rename persisted in DB | PASS | title=Renamed E2E Session |
| Delete session | PASS |  |
| Session gone after delete | PASS |  |
| Stream connected | PASS | status=200 |
| Received streaming tokens | PASS | chunk_count=50 |
| Assembled answer non-empty | PASS | len=256 |
| Stream finished with done event | PASS |  |
| Stream latency ok (<25s) | PASS | latency=2.85s |
