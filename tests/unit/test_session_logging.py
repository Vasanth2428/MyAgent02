import os
import shutil
import unittest
import logging
from src.core.logging_setup import SessionFileHandler, session_id_var

class TestSessionLogging(unittest.TestCase):
    def setUp(self):
        self.default_log = "logs/test_rag_engine.log"
        self.sessions_dir = "logs/test_sessions"
        
        # Ensure clean directories
        if os.path.exists(self.default_log):
            os.remove(self.default_log)
        if os.path.exists(self.sessions_dir):
            shutil.rmtree(self.sessions_dir)
            
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Setup Logger
        self.logger = logging.getLogger("TestSessionLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        self.formatter = logging.Formatter('%(message)s')
        self.handler = SessionFileHandler(
            default_log_path=self.default_log,
            log_dir=self.sessions_dir,
            formatter=self.formatter
        )
        self.logger.addHandler(self.handler)

    def tearDown(self):
        # Close handler properly before deleting files
        self.logger.removeHandler(self.handler)
        self.handler.close()
        
        # Clean up files
        if os.path.exists(self.default_log):
            os.remove(self.default_log)
        if os.path.exists(self.sessions_dir):
            shutil.rmtree(self.sessions_dir)

    def test_default_logging_no_session(self):
        """Verify logs are written to the default file when no session is active."""
        self.logger.info("Test message 1 - no session")
        
        # Verify default log contains message
        self.assertTrue(os.path.exists(self.default_log))
        with open(self.default_log, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Test message 1 - no session", content)
        
        # Verify no session log files are created
        session_files = os.listdir(self.sessions_dir)
        self.assertEqual(len(session_files), 0)

    def test_session_specific_logging(self):
        """Verify logs are written to both default and session-specific log files when session is active."""
        session_id = "test-session-xyz"
        token = session_id_var.set(session_id)
        
        try:
            self.logger.info("Test message 2 - with session")
        finally:
            session_id_var.reset(token)
            
        # Verify default log contains the message
        self.assertTrue(os.path.exists(self.default_log))
        with open(self.default_log, "r", encoding="utf-8") as f:
            default_content = f.read()
        self.assertIn("Test message 2 - with session", default_content)
        
        # Verify session log is created and contains the message
        session_log_path = os.path.join(self.sessions_dir, f"{session_id}.log")
        self.assertTrue(os.path.exists(session_log_path))
        with open(session_log_path, "r", encoding="utf-8") as f:
            session_content = f.read()
        self.assertIn("Test message 2 - with session", session_content)
