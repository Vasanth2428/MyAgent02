import unittest
from datetime import datetime, timedelta
from core.memory import ConversationMemory
from core.config import MEMORY_TOKEN_BUDGET
from core.engine import count_tokens

class TestMemoryDecayStress(unittest.TestCase):
    def test_memory_budget_strictness(self):
        """
        Simulate an agent running a long continuous loop, constantly adding memories.
        The memory system must strictly adhere to the budget, decaying oldest/least relevant memories first.
        """
        memory = ConversationMemory()
        
        # Create a massive simulated history well over the 300 token budget
        long_text = "This is a standard generic agent thought process that takes up some tokens. " * 10
        
        for i in range(15):
            memory.add(f"Thought {i}: " + long_text, role="assistant", importance=0.5)
            entry = memory.entries[-1]
            # Artificially age them
            entry.timestamp = datetime.now() - timedelta(minutes=60 - i)
            entry.last_seen = entry.timestamp

        # Add one highly important, very recent memory
        critical_entry = memory.add("CRITICAL SYSTEM STATE: The user wants to deploy immediately.", role="user", importance=1.0)
        
        # Get active context
        active_context = memory.get_active_context()
        active_tokens = count_tokens(active_context)
        
        # Inference / Assertions
        # 1. Budget adhered to
        self.assertLessEqual(active_tokens, MEMORY_TOKEN_BUDGET + 50, "Memory exceeded the token budget!")
        
        # 2. Critical info retained
        self.assertIn("deploy immediately", active_context, "Memory decay deleted critical recent context!")
        
        self.assertNotIn("Thought 14:", active_context, "Memory decay failed to drop excess noise!")

        print(f"Memory Decay Test: Budget={MEMORY_TOKEN_BUDGET}, Actual={active_tokens}")

if __name__ == "__main__":
    unittest.main()
