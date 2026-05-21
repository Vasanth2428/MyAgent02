import subprocess
import os
import sys
import time

# Ensure we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(path):
    print(f"\nRUNNING: {path}")
    try:
        subprocess.run([sys.executable, path], check=True)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== MODULAR RAG SUITE TEST ===\n")
    
    # 1. Start the new main.py in the background
    print("Starting Modular API...")
    server = subprocess.Popen([sys.executable, "main.py"])
    
    # Wait for the server to be ready
    import urllib.request
    print("Waiting for server to become responsive...")
    max_retries = 30
    server_ready = False
    for i in range(max_retries):
        try:
            with urllib.request.urlopen("http://localhost:8000/stats", timeout=2) as response:
                if response.status == 200:
                    print("Server is ready!")
                    server_ready = True
                    break
        except Exception:
            pass
        time.sleep(2)
        print(f"Retrying connection ({i+1}/{max_retries})...")
        
    if not server_ready:
        print("Error: Server failed to start in time.")
        server.terminate()
        sys.exit(1)
    
    try:
        # 2. Run integration tests
        run_script("tests/integration/verify_api.py")
        run_script("tests/integration/test_dual_channel.py")
    finally:
        print("Shutting down server...")
        server.terminate()
