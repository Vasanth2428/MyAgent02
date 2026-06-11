# Real End-to-End System Test

*Generated on: 2026-06-11T10:15:31.125143*

**Overall: ALL PASS**

| Check | Result | Detail |
|---|---|---|
| Server responding | PASS |  |
| Engine online | PASS | status=online |
| Documents indexed | PASS | count=2 |
| CPU readable | PASS | cpu=14.8% |
| Request succeeded | PASS | status=200 |
| Got a non-empty answer | PASS | len=284 chars |
| Answer mentions core concepts | PASS | Answer: I couldn't find any information about the 1% rule in Atomic Habits or ho |
| LLM used context (not hallucinating) | PASS | retrieved_context present: True |
| Latency acceptable (<20s) | PASS | latency=5.77s |
| Query count incremented | PASS | queries=7 |
| First IKIGAI query succeeded | PASS |  |
| Answer mentions Ikigai concept | PASS | snippet: Unfortunately, there is no information provided about the Japanese conc |
| Follow-up query succeeded | PASS |  |
| Follow-up answer is non-empty | PASS | len=213 |
| Follow-up references context | PASS | snippet: Unfortunately, I don't have any information about the Japanese concept  |
| History persisted in DB | PASS | history entries=4 |
| SQL aggregation returns data | PASS | result snippet: | status | count | total |
|--------|-------|-------|
| Shipped  |
| Customer count query works | PASS | result: | total |
|-------|
| 8 | |
| DROP blocked | PASS | response: Error: Only SELECT queries are permitted. The keyword 'DROP' is forbid |
| UNION blocked | PASS | response: Error: Only SELECT queries are permitted. The keyword 'UNION' is forbi |
| Web search returned results | PASS | count=5 |
| Results have real URLs (not mock) | PASS | urls=['https://www.itpro.com/technology/artificial-intelligence/wha', 'https://i |
| Results have real content | PASS | content lengths=[145, 152, 150, 141, 151] |
| Latency under 10s | PASS | latency=5.10s |
| Create session | PASS | status=201 |
| Session ID returned | PASS | sid=SID-3785B746 |
| List sessions works | PASS |  |
| New session in list | PASS | session count=9 |
| Rename session | PASS |  |
| Rename persisted in DB | PASS | title=Renamed E2E Session |
| Delete session | PASS |  |
| Session gone after delete | PASS |  |
| Stream connected | PASS | status=200 |
| Received streaming tokens | PASS | chunk_count=50 |
| Assembled answer non-empty | PASS | len=256 |
| Stream finished with done event | PASS |  |
| Stream latency ok (<25s) | PASS | latency=3.16s |
