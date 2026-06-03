import sys
import os
import argparse
import pytest

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    border = "=" * 65
    print("\n" + border)
    print(f" {title.center(63)}")
    print(border + "\n")

def check_server_running():
    """Verify if local FastAPI port 8000 is open before running integration tests."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(("127.0.0.1", 8000))
            return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Consolidated RAG Test Suite Runner")
    parser.add_argument(
        "suite",
        choices=["all", "unit", "integration", "multi_agent", "diagnostics", "stress"],
        default="all",
        nargs="?",
        help="Specify the test suite category to run (default: %(default)s)"
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Force skip integration tests even when running 'all'"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose pytest output"
    )
    
    args = parser.parse_args()
    
    print_header("RAG ENGINE - UNIFIED TEST SUITE RUNNER")
    
    pytest_args = []
    if args.verbose:
        pytest_args.append("-v")
        
    suites_to_run = []
    
    if args.suite == "all":
        suites_to_run.extend(["unit", "multi_agent", "diagnostics", "stress"])
        
        # Check integration server
        if not args.skip_integration:
            if check_server_running():
                print("[INFO] Local server running on port 8000 detected. Including integration tests.")
                suites_to_run.append("integration")
            else:
                print("[WARNING] Local server not running on port 8000. Skipping integration tests.")
        else:
            print("[INFO] Integration tests explicitly skipped.")
    else:
        suites_to_run.append(args.suite)
        
    paths = []
    for suite in suites_to_run:
        path = f"tests/{suite}"
        if os.path.exists(path):
            paths.append(path)
        else:
            print(f"[WARNING] Path '{path}' not found, skipping.")
            
    if not paths:
        print("[ERROR] No valid test paths resolved. Exiting.")
        sys.exit(1)
        
    print(f"[RUNNING] Executing categories: {', '.join(suites_to_run)}")
    print(f"[PATHS] Target paths: {', '.join(paths)}\n")
    
    pytest_args.extend(paths)
    
    # Run pytest programmatically
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 65)
    if exit_code == pytest.ExitCode.OK:
        print("               SUCCESS: ALL COMPLETED TESTS PASSED            ")
        print("=" * 65)
        sys.exit(0)
    else:
        print("               FAILURE: TEST SUITE FAILED WITH ERRORS         ")
        print("=" * 65)
        sys.exit(int(exit_code))


if __name__ == "__main__":
    main()
