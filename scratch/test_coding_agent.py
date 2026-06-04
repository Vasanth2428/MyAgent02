import sys
import os

# Ensure project imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.agents.coding_worker import coding_worker_node

def test_run():
    print("Starting agent test...")
    state = {
        "current_task": "Create a simple HTML/CSS webpage with index.html and style.css for a personal developer portfolio. It should have a clean design, a header, a projects section, and a contact section.",
        "scratchpad": "None",
        "messages": []
    }
    
    # Clean workspace folder first
    workspace_dir = os.path.join(os.path.dirname(__file__), "..", "workspace")
    print(f"Cleaning workspace folder: {workspace_dir}")
    if os.path.exists(workspace_dir):
        for item in os.listdir(workspace_dir):
            path = os.path.join(workspace_dir, item)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
                
    result = coding_worker_node(state)
    print("\n--- AGENT EXECUTION COMPLETED ---")
    print("Worker Output:")
    print(result.get("worker_outputs", {}).get("coding_worker", "No output"))
    
    # Check if files were created in the workspace
    print("\n--- FILES IN WORKSPACE ---")
    if os.path.exists(workspace_dir):
        files = os.listdir(workspace_dir)
        print(f"Files found: {files}")
    else:
        print("Workspace folder does not exist!")

if __name__ == "__main__":
    test_run()
