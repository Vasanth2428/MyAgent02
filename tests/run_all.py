import unittest
import sys
import os
import argparse

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests(directory, pattern="test_*.py"):
    if not os.path.exists(directory):
        print(f"Skipping {directory} (not found)")
        return True
        
    print(f"\n[{directory.upper()}]")
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=directory, pattern=pattern)
    
    if suite.countTestCases() == 0:
        print("No tests found.")
        return True
        
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Test Runner for RAG Engine")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests (which require API server)")
    args = parser.parse_args()

    print("============================================================")
    print("          RAG CONTEXT ENGINE - MASTER TEST SUITE            ")
    print("============================================================\n")

    success = True

    # 1. Unit Tests (Isolated module testing)
    success &= run_tests("tests/unit")

    # 2. Diagnostic / Introspective Tests (Deep capability evaluation)
    success &= run_tests("tests/diagnostics")

    # 3. Integration Tests (End-to-End API testing)
    if not args.skip_integration:
        print("\nNote: Integration tests assume 'python main.py' is running on port 8000.")
        success &= run_tests("tests/integration")
    else:
        print("\n[INTEGRATION] Skipped.")

    print("\n============================================================")
    if success:
        print("                ALL TEST SUITES PASSED                      ")
        sys.exit(0)
    else:
        print("                SOME TESTS FAILED                           ")
        sys.exit(1)
