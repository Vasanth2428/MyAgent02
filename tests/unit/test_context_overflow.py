import unittest
from unittest.mock import MagicMock, patch
from src.core.engine import RAGContextEngine

class TestContextOverflow(unittest.TestCase):

    def setUp(self):
        # Mock retriever to avoid database connections
        self.retriever = MagicMock()
        self.retriever.get_count.return_value = 10
        
        # Instantiate engine
        with patch('src.core.engine.LLMService') as mock_llm_service:
            mock_llm = MagicMock()
            mock_llm_service.return_value = mock_llm
            self.engine = RAGContextEngine(self.retriever)

    def test_handle_context_overflow_no_breach(self):
        """If the context size is below the limit, no overflow should occur."""
        query = "What is RAG?"
        final_context = "### MEMORY\n[user]: hi\n[assistant]: hello\n\n### KNOWLEDGE\n<document source=\"doc.txt\">\nThis is a short document.\n</document>"
        memory_text = "[user]: hi\n[assistant]: hello\n"
        compressed_docs = '<document source="doc.txt">\nThis is a short document.\n</document>'
        
        memory = MagicMock()
        memory.entries = [MagicMock(), MagicMock()]
        memory.get_active_context.return_value = memory_text
        
        all_raw = [{"text": "This is a short document.", "source": "doc.txt"}]
        
        # Limit is large
        limit = 4096
        
        (new_context, new_mem_text, new_docs, overflow_occurred, steps, 
         initial, final, mem_tkn, doc_tkn) = self.engine._handle_context_overflow(
            query, final_context, memory_text, compressed_docs, memory, all_raw, limit
        )
        
        self.assertFalse(overflow_occurred)
        self.assertEqual(len(steps), 0)
        self.assertEqual(new_context, final_context)

    def test_handle_context_overflow_prunes_memory(self):
        """If context is too large, it should prune old memory entries first."""
        query = "Explain memory decay."
        
        # Construct large memory text
        memory_entries = []
        for i in range(5):
            entry = MagicMock()
            entry.role = "user" if i % 2 == 0 else "assistant"
            entry.text = f"Turn number {i} containing a moderately long sentence."
            memory_entries.append(entry)
            
        memory_text = "".join([f"[{e.role}]: {e.text}\n" for e in memory_entries])
        compressed_docs = '<document source="doc.txt">\nShort doc.\n</document>'
        final_context = f"### MEMORY\n{memory_text}\n\n### KNOWLEDGE\n{compressed_docs}"
        
        memory = MagicMock()
        memory.entries = memory_entries
        # When active context is called, return whatever is in entries
        memory.get_active_context.side_effect = lambda: "".join([f"[{e.role}]: {e.text}\n" for e in memory.entries])
        
        all_raw = [{"text": "Short doc.", "source": "doc.txt"}]
        
        # Set a low limit where only about 1-2 entries can fit
        limit = 80
        
        (new_context, new_mem_text, new_docs, overflow_occurred, steps, 
         initial, final, mem_tkn, doc_tkn) = self.engine._handle_context_overflow(
            query, final_context, memory_text, compressed_docs, memory, all_raw, limit
        )
        
        self.assertTrue(overflow_occurred)
        self.assertTrue(any("Phase 1" in s for s in steps))
        self.assertLess(len(memory.entries), 5, "Memory entries should have been pruned")
        self.assertLess(mem_tkn, 100)

    @patch('src.core.compressor.Compressor.compress')
    def test_handle_context_overflow_compresses_knowledge(self, mock_compress):
        """If pruning memory isn't enough, it should compress documents aggressively."""
        query = "Explain retrieval."
        
        memory = MagicMock()
        memory.entries = [] # No memory turns to prune
        memory.get_active_context.return_value = ""
        memory_text = ""
        
        # Large initial document text
        compressed_docs = '<document source="doc.txt">\n' + ("This is a long text chunk. " * 30) + '\n</document>'
        final_context = f"### MEMORY\n\n\n### KNOWLEDGE\n{compressed_docs}"
        
        all_raw = [{"text": "This is a long text chunk. " * 30, "source": "doc.txt"}]
        
        # Mock compressor output to be small
        mock_compress.return_value = "Compressed."
        
        # Low limit
        limit = 160
        
        (new_context, new_mem_text, new_docs, overflow_occurred, steps, 
         initial, final, mem_tkn, doc_tkn) = self.engine._handle_context_overflow(
            query, final_context, memory_text, compressed_docs, memory, all_raw, limit
        )
        
        self.assertTrue(overflow_occurred)
        self.assertTrue(any("Phase 2" in s for s in steps))
        mock_compress.assert_called_once()
        self.assertIn("Compressed.", new_docs)

if __name__ == "__main__":
    unittest.main()
