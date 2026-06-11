import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from src.core.code.parser import parse_code_file
from src.core.code.symbol_table import SymbolTable
from src.core.code.dependency_graph import DependencyGraph
from src.core.code.indexer import CodeIndexer
from src.core.code.code_registry import CodeRegistry
from src.tools.patch_tools import generate_diff_patch, apply_patch, dry_run_patch
from src.core.code.validation import validate_syntax
from src.agents.code_critic_worker import code_critic_worker_node

# Sample python code to write into temporary files for testing
SAMPLE_CODE = '''"""Module docstring."""
import math
from os import path

class DataProcessor:
    """Class docstring."""
    def __init__(self, value: int) -> None:
        self.value = value

    def process(self) -> float:
        """Method docstring."""
        return math.sqrt(self.value)

def helper_function(x: float) -> str:
    """Helper function."""
    processor = DataProcessor(int(x))
    res = processor.process()
    return f"Result: {res}"
'''

@pytest.fixture
def temp_py_file():
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_CODE)
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_parser(temp_py_file):
    res = parse_code_file(temp_py_file)
    assert not res.get("error")
    
    symbols = res["symbols"]
    classes = [s for s in symbols if s["type"] == "class"]
    methods = [s for s in symbols if s["type"] == "method"]
    functions = [s for s in symbols if s["type"] == "function"]
    
    assert len(classes) == 1
    assert classes[0]["name"] == "DataProcessor"
    assert classes[0]["docstring"] == "Class docstring."
    assert classes[0]["methods"] == ["__init__", "process"]
    
    assert len(methods) == 2
    assert methods[0]["name"] == "__init__"
    assert methods[1]["name"] == "process"
    
    assert len(functions) == 1
    assert functions[0]["name"] == "helper_function"
    assert functions[0]["return_type"] == "str"
    
    imports = res["imports"]
    assert len(imports) == 2
    assert imports[0]["name"] == "math"
    assert imports[1]["module"] == "os"
    assert imports[1]["name"] == "path"


def test_symbol_table():
    tbl = SymbolTable()
    sym = {
        "type": "class",
        "name": "TestClass",
        "filepath": "test.py",
        "start_line": 5,
        "end_line": 20,
        "docstring": "Test class"
    }
    tbl.add_symbol(sym)
    
    assert len(tbl.get_symbols_by_name("TestClass")) == 1
    assert tbl.get_symbol("TestClass")["filepath"] == "test.py"
    
    fuzzy = tbl.search_symbols("testcl")
    assert len(fuzzy) == 1
    assert fuzzy[0]["name"] == "TestClass"
    
    tbl.clear()
    assert len(tbl.get_all_symbols()) == 0


def test_dependency_graph():
    graph = DependencyGraph()
    graph.add_import("main.py", "src.core")
    graph.add_import("main.py", "weaviate")
    graph.add_call("main.py", "WeaviateRetriever")
    
    assert "src.core" in graph.get_imports("main.py")
    assert "weaviate" in graph.get_imports("main.py")
    assert "main.py" in graph.get_imported_by("weaviate")
    assert "WeaviateRetriever" in graph.get_callees("main.py")
    assert "main.py" in graph.get_callers("WeaviateRetriever")
    
    graph.clear()
    assert len(graph.get_imports("main.py")) == 0


def test_indexer_and_registry():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test python file inside the temporary directory
        file_path = os.path.join(temp_dir, "test_module.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CODE)
            
        indexer = CodeIndexer(temp_dir)
        indexer.index_repository()
        
        assert "test_module.py" in indexer.indexed_files
        assert len(indexer.symbol_table.get_all_symbols()) > 0
        
        # Test serialization using registry
        index_json = os.path.join(temp_dir, "index.json")
        registry = CodeRegistry(index_json, indexer)
        registry.save_index()
        
        assert os.path.exists(index_json)
        
        # Reset and load
        indexer2 = CodeIndexer(temp_dir)
        registry2 = CodeRegistry(index_json, indexer2)
        assert registry2.load_index()
        
        assert "test_module.py" in indexer2.indexed_files
        assert len(indexer2.symbol_table.get_all_symbols()) > 0


def test_patch_tools():
    original = "line 1\nline 2\nline 3\n"
    modified = "line 1\nline 2 modified\nline 3\n"
    
    diff = generate_diff_patch("test.py", original, modified)
    assert "line 2 modified" in diff
    
    patched = apply_patch(original, diff)
    assert patched == modified
    
    # Mismatch check
    bad_diff = diff.replace("line 1", "line 1 bad")
    with pytest.raises(ValueError):
        apply_patch(original, bad_diff)


def test_validation():
    valid_code = "def add(a, b): return a + b"
    invalid_code = "def add(a, b) return a + b" # Missing colon
    
    ok, msg = validate_syntax(valid_code, "test.py")
    assert ok is True
    
    ok_fail, msg_fail = validate_syntax(invalid_code, "test.py")
    assert ok_fail is False
    assert "Syntax Error" in msg_fail


@patch("src.agents.code_critic_worker.get_critic_model")
@patch("src.agents.coding_worker.get_retrieval_service")
def test_critic_node(mock_service, mock_get_critic):
    # Mock symbols list from service
    mock_indexer = MagicMock()
    mock_indexer.symbol_table.get_all_symbols.return_value = [
        {"name": "valid_symbol", "type": "function", "filepath": "main.py", "start_line": 1, "end_line": 5}
    ]
    mock_srv = MagicMock(indexer=mock_indexer)
    mock_service.return_value = mock_srv
    
    # Mock LLM Critic Report
    mock_report = MagicMock(
        valid=True,
        findings=[],
        criticism_summary="All code looks correct and fully validated."
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_report
    mock_get_critic.return_value = mock_llm
    
    state = {
        "scratchpad": "",
        "current_task": "Write coding task",
        "worker_outputs": {
            "coding_worker": "Created script implementing valid_symbol."
        }
    }
    
    res = code_critic_worker_node(state)
    assert res["worker_complete"]["code_critic_worker"] is True
    assert "CODE CRITIC VALIDATION REPORT" in res["worker_outputs"]["code_critic_worker"]
    assert "valid_symbol" in res["scratchpad"] or "validated" in res["scratchpad"]


@patch("src.agents.code_critic_worker.get_critic_model")
@patch("src.agents.coding_worker.get_retrieval_service")
def test_critic_node_retry_required(mock_service, mock_get_critic):
    # Mock symbols list from service
    mock_indexer = MagicMock()
    mock_indexer.symbol_table.get_all_symbols.return_value = []
    mock_srv = MagicMock(indexer=mock_indexer)
    mock_service.return_value = mock_srv
    
    # Mock LLM Critic Report with critical failure
    mock_finding = MagicMock(
        issue_type="hallucinated_symbol",
        symbol_name="missing_func",
        file_location="main.py",
        details="Function missing_func does not exist in repository.",
        severity="critical",
        evidence="Checked symbol table."
    )
    mock_report = MagicMock(
        valid=False,
        findings=[mock_finding],
        criticism_summary="Found critical symbol hallucination."
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_report
    mock_get_critic.return_value = mock_llm
    
    state = {
        "scratchpad": "",
        "current_task": "Write coding task",
        "plan": ["Write code"],
        "worker_outputs": {
            "coding_worker": "Created script implementing missing_func."
        }
    }
    
    res = code_critic_worker_node(state)
    assert res["worker_complete"]["code_critic_worker"] is True
    assert "RETRY_REQUIRED" in res["worker_outputs"]["code_critic_worker"]
    # Check that the plan was updated with a fix instruction that includes the critic feedback
    assert res["plan"][-1].startswith("FIX: ")
    # The fix instruction should contain the symbol name and file location from the critic finding
    assert "missing_func" in res["plan"][-1]
    assert "main.py" in res["plan"][-1]


def test_security_audit():
    """Test security audit functionality for vulnerability detection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with vulnerabilities
        vuln_file = os.path.join(temp_dir, "vulnerable.py")
        vuln_code = '''
PASSWORD = "super_secret_password_123"
api_key = "sk-12345"
def unsafe_sql(user):
    cursor.execute(f"SELECT * FROM users WHERE name='{user}'")
def unsafe_cmd(cmd):
    import subprocess
    subprocess.call(f"echo {cmd}", shell=True)  # f-string triggers pattern
'''
        with open(vuln_file, "w", encoding="utf-8") as f:
            f.write(vuln_code)
        
        # Mock retriever to avoid Weaviate connection
        from src.core.services.code_retrieval_service import CodeRetrievalService
        from unittest.mock import MagicMock
        
        mock_retriever = MagicMock()
        service = CodeRetrievalService(temp_dir, mock_retriever)
        
        findings = service.audit_security("vulnerable.py")
        
        assert "hardcoded_secret" in findings
        assert len(findings["hardcoded_secret"]) >= 1
        assert any("PASSWORD" in str(f) for f in findings["hardcoded_secret"])
        
        assert "sql_injection" in findings
        assert len(findings["sql_injection"]) >= 1
        
        assert "command_injection" in findings
        assert len(findings["command_injection"]) >= 1


def test_call_graph():
    """Test call graph analysis for dependency tracking."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        main_file = os.path.join(temp_dir, "main.py")
        with open(main_file, "w", encoding="utf-8") as f:
            f.write('''
def main():
    helper()
def helper():
    pass
''')
        
        from src.core.services.code_retrieval_service import CodeRetrievalService
        from unittest.mock import MagicMock
        
        mock_retriever = MagicMock()
        service = CodeRetrievalService(temp_dir, mock_retriever)
        
        graph = service.get_call_graph()
        assert "calls" in graph
        assert "called_by" in graph


def test_tree_sitter_parser():
    """Test Tree-sitter parser for robust code extraction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test Python file
        test_file = os.path.join(temp_dir, "ts_test.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(SAMPLE_CODE)
        
        # Test with explicit tree-sitter language
        res = parse_code_file(test_file, language="python")
        
        if not res.get("error"):
            # Tree-sitter is available, check results
            symbols = res["symbols"]
            classes = [s for s in symbols if s["type"] == "class"]
            methods = [s for s in symbols if s["type"] in ("method", "async_function")]
            functions = [s for s in symbols if s["type"] == "function"]
            
            assert len(classes) >= 1
            assert any(c["name"] == "DataProcessor" for c in classes)
            assert len(methods) >= 2 or len(functions) >= 1
        else:
            # Tree-sitter not installed, skip
            pytest.skip(f"Tree-sitter not available: {res.get('error')}")


def test_multi_language_indexing():
    """Test indexing multiple file types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Python file
        py_file = os.path.join(temp_dir, "module.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("def test(): pass\n")
        
        indexer = CodeIndexer(temp_dir)
        indexer.index_repository()
        
        # Python file should be indexed
        assert "module.py" in indexer.indexed_files
