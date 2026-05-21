import unittest
import requests

API_URL = "http://localhost:8000/upload"

class TestAPIUpload(unittest.TestCase):
    def test_upload_txt_file(self):
        files = {'file': ('test_doc.txt', 'This is a test document for semantic chunking and indexing.')}
        response = requests.post(API_URL, files=files)
        
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        data = response.json()
        self.assertEqual(data.get("status"), "success")
        self.assertIn("Indexed", data.get("message", ""))

if __name__ == "__main__":
    unittest.main()
