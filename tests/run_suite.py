import unittest
import sys
import os

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("============================================================")
    print("          RAG CONTEXT ENGINE - PRODUCTION TEST SUITE        ")
    print("============================================================\n")

    # Discover and load tests in tests/integration
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests/integration", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n============================================================")
    print("                TEST SUITE EXECUTION FINISHED               ")
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print("============================================================")

    # Return non-zero code on failures so CI/CD processes can detect them
    if not result.wasSuccessful():
        sys.exit(1)
