import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("============================================================")
    print("          MULTI-AGENT SYSTEM - SAFETY TEST SUITE          ")
    print("============================================================")
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests/multi_agent", pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n============================================================")
    print("                TEST SUITE EXECUTION FINISHED               ")
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print("============================================================")
    
    if not result.wasSuccessful():
        sys.exit(1)
