import unittest
import requests
import json
import time

BASE_URL = "http://localhost:8000"

class TestFrontendBackendStreaming(unittest.TestCase):
    def test_frontend_assets_served(self):
        """Verify that index.html, static/app.js, and static/styles.css are served properly."""
        # 1. Test index.html
        resp_html = requests.get(f"{BASE_URL}/")
        self.assertEqual(resp_html.status_code, 200, f"Failed to load /: {resp_html.status_code}")
        self.assertIn("RAG Context Hub & Console", resp_html.text, "index.html content mismatch")

        # 2. Test static/app.js
        resp_js = requests.get(f"{BASE_URL}/static/app.js")
        self.assertEqual(resp_js.status_code, 200, f"Failed to load app.js: {resp_js.status_code}")
        self.assertIn("API_BASE", resp_js.text, "app.js content mismatch")

        # 3. Test static/styles.css
        resp_css = requests.get(f"{BASE_URL}/static/styles.css")
        self.assertEqual(resp_css.status_code, 200, f"Failed to load styles.css: {resp_css.status_code}")

    def test_agentic_streaming(self):
        """Verify multi-agent execution and the SSE stream protocol events."""
        payload = {
            "question": "Calculate 45 * 78 and give the final result.",
            "session_id": "test-frontend-agent-stream",
            "mode": "agentic",
            "context_limit": 16384
        }
        
        events_received = []
        chunks = []
        has_done = False
        
        with requests.post(f"{BASE_URL}/query_stream", json=payload, stream=True, timeout=30) as resp:
            self.assertEqual(resp.status_code, 200, f"Streaming request failed: {resp.status_code}")
            
            for line in resp.iter_lines():
                if not line:
                    continue
                raw_line = line.decode("utf-8")
                if raw_line.startswith("data: "):
                    json_str = raw_line[6:].strip()
                    if not json_str:
                        continue
                    try:
                        data = json.loads(json_str)
                        events_received.append(data)
                        
                        event_type = data.get("event")
                        if event_type == "answer_chunk":
                            chunks.append(data.get("text", ""))
                        elif event_type == "done":
                            has_done = True
                    except Exception as e:
                        print(f"Failed to parse SSE payload: {json_str}. Error: {e}")

        # Validate that we received streaming events
        self.assertGreater(len(events_received), 0, "No SSE events received")
        
        # Verify that we received specific multi-agent orchestration events
        event_types = [ev.get("event") for ev in events_received]
        
        # We expect thoughts/actions from routing & execution or answer chunks
        self.assertTrue(
            any(et in ["thought", "action", "node_start", "state_change"] for et in event_types),
            f"Expected agent orchestration events in: {event_types}"
        )
        
        # Verify that answer chunks were streamed
        full_answer = "".join(chunks)
        self.assertGreater(len(full_answer), 0, "No answer chunks streamed")
        self.assertTrue(has_done, "Stream did not finish with a 'done' event")
        
        print(f"\n[STREAM TEST] Received {len(events_received)} SSE events.")
        print(f"[STREAM TEST] Assembled answer: {full_answer}")

if __name__ == "__main__":
    unittest.main()
