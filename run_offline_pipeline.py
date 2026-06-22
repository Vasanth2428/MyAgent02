import os
import sys
import asyncio
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

# Load active environment variables
load_dotenv("config/.env", override=True)

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.graph.workflow import build_multi_agent_graph, get_graph_config

# Mock the LLM models to simulate the agentic pipeline execution offline
class PredefinedSupervisor:
    def __init__(self):
        self.call_count = 0
        
    def invoke(self, messages, *args, **kwargs):
        self.call_count += 1
        from src.graph.supervisor import SupervisorDecision
        
        if self.call_count == 1:
            # First, supervisor routes to coding worker to scaffold the app
            return SupervisorDecision(
                plan=["Scaffold frontend", "Write backend API", "Build and verify"],
                next_agent="coding_worker",
                current_task="Scaffold the React frontend app named 'crypto_portfolio' inside `./workspace`"
            )
        elif self.call_count == 2:
            # Second step, supervisor routes to coding worker to build backend
            return SupervisorDecision(
                plan=["Write backend API", "Build and verify"],
                next_agent="coding_worker",
                current_task="Create backend files: main.py, database, requirements.txt under `./workspace/crypto_portfolio/backend/`"
            )
        elif self.call_count == 3:
            # Third step, supervisor routes to coding worker to write frontend components
            return SupervisorDecision(
                plan=["Write frontend App.jsx & App.css", "Build and verify"],
                next_agent="coding_worker",
                current_task="Create frontend code: App.jsx and App.css under `./workspace/crypto_portfolio/src/`"
            )
        elif self.call_count == 4:
            # Fourth step, build and verify
            return SupervisorDecision(
                plan=["Build and verify"],
                next_agent="coding_worker",
                current_task="Verify backend syntax and build the React frontend using npm run build"
            )
        else:
            # Final step, route to synthesizer
            return SupervisorDecision(
                plan=[],
                next_agent="synthesizer",
                current_task=""
            )

class PredefinedCodingWorker:
    def __init__(self):
        self.call_count = 0
        
    def invoke(self, messages, *args, **kwargs):
        self.call_count += 1
        
        # We simulate the tool calls that the coding worker would make
        if self.call_count == 1:
            # Scaffolds the app
            return AIMessage(
                content="I will scaffold the React frontend.",
                tool_calls=[{
                    "name": "scaffold_react_app",
                    "args": {"project_name": "crypto_portfolio"},
                    "id": "scaffold_call"
                }]
            )
        elif self.call_count == 2:
            # Scaffold returns success. Now write the backend files.
            backend_code = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlite3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/prices")
def get_prices():
    return {
        "status": "success",
        "prices": [
            {"name": "Bitcoin", "symbol": "BTC", "price": 68450.20, "change_24h": 2.4},
            {"name": "Ethereum", "symbol": "ETH", "price": 3520.15, "change_24h": -1.2},
            {"name": "Solana", "symbol": "SOL", "price": 145.80, "change_24h": 5.7}
        ]
    }
"""
            # We return two tool calls: create main.py and create requirements.txt
            return AIMessage(
                content="I will create the backend FastAPI script.",
                tool_calls=[{
                    "name": "create_files",
                    "args": {
                        "filepath": "crypto_portfolio/backend/main.py",
                        "content": backend_code
                    },
                    "id": "backend_call_1"
                }]
            )
        elif self.call_count == 3:
            # Now create frontend dashboard code
            app_jsx = """import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [prices, setPrices] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In local dev, fetch mock prices
    setTimeout(() => {
      setPrices([
        {name: "Bitcoin", symbol: "BTC", price: 68450.20, change_24h: 2.4},
        {name: "Ethereum", symbol: "ETH", price: 3520.15, change_24h: -1.2},
        {name: "Solana", symbol: "SOL", price: 145.80, change_24h: 5.7}
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1 className="title">Crypto Portfolio Hub</h1>
        {loading ? (
          <p>Loading prices...</p>
        ) : (
          <div className="grid">
            {prices.map(p => (
              <div key={p.symbol} className="card">
                <h3>{p.name} ({p.symbol})</h3>
                <p className="price">${p.price.toLocaleString()}</p>
                <p className={p.change_24h >= 0 ? "change up" : "change down"}>
                  {p.change_24h >= 0 ? "+" : ""}{p.change_24h}%
                </p>
              </div>
            ))}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;"""

            app_css = """body {
  margin: 0;
  font-family: 'Outfit', 'Inter', sans-serif;
  background-color: #0b0f19;
  color: #f3f4f6;
}
.App {
  text-align: center;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
.title {
  font-size: 2.5rem;
  background: linear-gradient(135deg, #60a5fa, #c084fc);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 2rem;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  max-width: 900px;
  width: 100%;
  padding: 20px;
}
.card {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  padding: 24px;
  backdrop-filter: blur(10px);
  transition: transform 0.2s;
}
.card:hover {
  transform: translateY(-5px);
}
.price {
  font-size: 1.5rem;
  font-weight: bold;
}
.change {
  font-weight: 600;
}
.change.up { color: #34d399; }
.change.down { color: #f87171; }
"""
            # Create/modify files App.jsx and App.css
            return AIMessage(
                content="I will write the frontend application code.",
                tool_calls=[
                    {
                        "name": "modify_files",
                        "args": {
                            "filepath": "crypto_portfolio/src/App.jsx",
                            "target_code": "",
                            "replacement_code": app_jsx
                        },
                        "id": "frontend_call_1"
                    },
                    {
                        "name": "modify_files",
                        "args": {
                            "filepath": "crypto_portfolio/src/App.css",
                            "target_code": "",
                            "replacement_code": app_css
                        },
                        "id": "frontend_call_2"
                    }
                ]
            )
        elif self.call_count == 4:
            # Build and verify using run_safe_commands
            return AIMessage(
                content="I will run syntax checks and build tests.",
                tool_calls=[
                    {
                        "name": "run_safe_commands",
                        "args": {"command": "python -m py_compile crypto_portfolio/backend/main.py"},
                        "id": "verify_call_1"
                    }
                ]
            )
        else:
            return AIMessage(content="Everything completed and verified.")

def main():
    print("Initializing Multi-Agent Workflow...")
    # Initialize graph checkpointer
    from src.graph.checkpointer import setup_checkpointer
    checkpointer = setup_checkpointer()
    graph = build_multi_agent_graph(checkpointer)
    
    # We patch the supervisor and coding worker models
    supervisor_mock = PredefinedSupervisor()
    coding_worker_mock = PredefinedCodingWorker()
    
    # Simple query
    query = "Create a fullstack crypto portfolio website inside `./workspace` named 'crypto_portfolio'"
    import time
    config = get_graph_config(f"offline_build_{int(time.time())}")
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "supervisor",
        "steps_remaining": 15,
        "plan": [],
        "current_task": "",
        "worker_complete": {},
        "retry_counter": 0,
        "critic_retry_count": 0,
        "waiting_for_approval": False,
        "approval_filepath": "",
        "pending_file_approvals": {},
        "patch_is_verified": False,
        "active_project": "crypto_portfolio",
        "session_id": "offline_build_session",
        "active_document_ids": [],
        "task_hashes": [],
        "file_status_flags": {},
        "worker_output_ids": {},
        "worker_output_summaries": {},
        "scratchpad_references": [],
        "scratchpad": "",
        "worker_outputs": {},
        "final_answer": "",
        "bypass_hitl": True # Bypass HITL for non-interactive runner
    }
    
    print("\nRunning Multi-Agent pipeline...")
    with patch("src.graph.supervisor.get_routing_model") as mock_super_model, \
         patch("src.agents.coding_worker.get_coding_model") as mock_coding_model:
         
         # Return our pre-defined mocks
         mock_super_llm = MagicMock()
         mock_super_llm.invoke.side_effect = supervisor_mock.invoke
         mock_super_model.return_value = mock_super_llm
         
         mock_coding_llm = MagicMock()
         mock_coding_llm.invoke.side_effect = coding_worker_mock.invoke
         mock_coding_model.return_value = mock_coding_llm
         
         result = graph.invoke(initial_state, config=config)
         print("\nPipeline run completed successfully.")

if __name__ == "__main__":
    main()
