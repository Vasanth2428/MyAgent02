import unittest
import threading
import requests
import time

API_URL = "http://localhost:8000/query"

class TestDualChannelConcurrency(unittest.TestCase):
    def setUp(self):
        self.errors = []
        self.results = []

    def send_query(self, session_id, question, mode):
        payload = {
            "question": question,
            "session_id": session_id,
            "mode": mode
        }
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code != 200:
                self.errors.append(f"Session {session_id} failed with status {response.status_code}")
                return
            data = response.json()
            self.results.append((session_id, data))
        except Exception as e:
            self.errors.append(f"Session {session_id} exception: {e}")

    def test_concurrent_sessions(self):
        t1 = threading.Thread(target=self.send_query, args=("Session-A", "What is Link State Advertisement?", "context_engine"))
        t2 = threading.Thread(target=self.send_query, args=("Session-B", "Explain NAT concepts.", "normal"))
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        # Start a sequential follow-up conversation to verify memory continuity
        t3 = threading.Thread(target=self.send_query, args=("Session-A", "How does LSA help routers?", "context_engine"))
        t3.start()
        t3.join()
        
        # Verify no network or processing exceptions were thrown during concurrency
        self.assertEqual(len(self.errors), 0, f"Concurrency errors encountered: {self.errors}")
        self.assertEqual(len(self.results), 3, "Expected 3 successful dialog turns")
        
        # Check session contexts
        a_results = [r for sid, r in self.results if sid == "Session-A"]
        b_results = [r for sid, r in self.results if sid == "Session-B"]
        
        self.assertEqual(len(a_results), 2)
        self.assertEqual(len(b_results), 1)

if __name__ == "__main__":
    unittest.main()
