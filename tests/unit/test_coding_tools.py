import os
import pytest
from src.tools.coding_tools import (
    read_files,
    search_code,
    create_files,
    modify_files,
    list_files,
    run_safe_commands,
    _is_safe_path,
    WORKSPACE_ROOT
)

TEST_FILE = "temp_test_file.txt"
TEST_ABS_PATH = os.path.join(WORKSPACE_ROOT, TEST_FILE)
TEST_CONTENT = """# Mock test file
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Create a temporary test file in the workspace folder
    os.makedirs(WORKSPACE_ROOT, exist_ok=True)
    with open(TEST_ABS_PATH, "w", encoding="utf-8") as f:
        f.write(TEST_CONTENT)
    yield
    # Teardown: Remove the file if it exists
    if os.path.exists(TEST_ABS_PATH):
        os.remove(TEST_ABS_PATH)


def test_is_safe_path():
    assert _is_safe_path("temp_test_file.txt") is True
    assert _is_safe_path("src/tools/coding_tools.py") is False  # outside workspace
    # Test path traversal containment
    assert _is_safe_path("../outside_workspace.txt") is False
    assert _is_safe_path("dir/../../passwd") is False
    assert _is_safe_path("etc/passwd") is True  # relative path is fine since it resolves to ./workspace/etc/passwd
    assert _is_safe_path("/etc/passwd") is False  # absolute path traversal /etc/ is blocked
    assert _is_safe_path("/") is False  # root blocked
    assert _is_safe_path("/usr") is False
    assert _is_safe_path("/root") is False
    assert _is_safe_path("/var") is False


def test_read_files():
    res = read_files(TEST_FILE, start_line=2, end_line=3)
    assert "def add(a, b):" in res
    assert "return a + b" in res
    assert "subtract" not in res
    
    # Test forbidden extension (e.g. .sh is not allowed under safety policy)
    res_sh = read_files("test.sh", start_line=1, end_line=10)
    assert "extension not allowed" in res_sh


def test_search_code():
    res = search_code("def subtract")
    assert TEST_FILE in res
    assert "subtract" in res


def test_create_and_modify_files():
    # Test create_files
    new_file = "new_created_file.txt"
    new_abs_path = os.path.join(WORKSPACE_ROOT, new_file)
    try:
        res_create = create_files(new_file, "Line 1\nLine 2")
        assert "Success" in res_create
        assert os.path.exists(new_abs_path)
        
        # Test creating existing file fails
        res_fail = create_files(new_file, "Different content")
        assert "Error" in res_fail
        
        # Test modify_files
        res_modify = modify_files(new_file, "Line 2", "Line 2 modified")
        assert "Success" in res_modify
        with open(new_abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Line 2 modified" in content
        
        # Test modifying non-existent file fails
        res_mod_fail = modify_files("non_existent_file.txt", "target", "replacement")
        assert "Error" in res_mod_fail
    finally:
        if os.path.exists(new_abs_path):
            os.remove(new_abs_path)


def test_list_files():
    res = list_files(".")
    assert TEST_FILE in res
    assert "[FILE]" in res


def test_run_safe_commands():
    # Test safe allowed command (starts with pytest)
    res = run_safe_commands("pytest --version")
    assert "pytest" in res.lower()
    
    # Test blocked command prefix (not in allowlist)
    res_blocked = run_safe_commands("python -c \"print('Hello')\"")
    assert "blocked" in res_blocked
    
    # Test blocked keyword in command (denylist)
    res_dangerous = run_safe_commands("pytest rm -rf /")
    assert "blocked" in res_dangerous or "forbidden" in res_dangerous


def test_symlink_traversal_mocked(monkeypatch):
    original_realpath = os.path.realpath
    def mock_realpath(path):
        p_str = str(path).replace("\\", "/")
        if "symlink_attack_file" in p_str:
            return original_realpath(os.path.join(WORKSPACE_ROOT, "..", "outside_file.txt"))
        return original_realpath(path)
    
    monkeypatch.setattr(os.path, "realpath", mock_realpath)
    assert _is_safe_path("symlink_attack_file.txt") is False


def test_package_json_blocks():
    assert _is_safe_path("package.json") is False
    assert _is_safe_path("dir/package.json") is False
    assert _is_safe_path("dir/../package.json") is False
    
    # Verify read/write functions reject package.json
    assert "violates safety" in read_files("package.json")
    assert "violates safety" in create_files("package.json", "{}")
    assert "violates safety" in modify_files("package.json", "a", "b")


def test_command_injection_blocks():
    assert "blocked" in run_safe_commands("pytest; rm -rf /")
    assert "blocked" in run_safe_commands("pytest && rm -rf /")
    assert "blocked" in run_safe_commands("pytest | rm -rf /")
    assert "blocked" in run_safe_commands("pytest $(echo rm)")
    assert "blocked" in run_safe_commands("pytest `echo rm`")
    assert "blocked" in run_safe_commands("python -c \"import os; os.system('echo')\"")


def test_resource_limits_timeout(monkeypatch):
    import time
    # Temporarily allow the test command
    monkeypatch.setattr("src.tools.coding_tools._is_safe_command", lambda args: True)
    
    start = time.time()
    res = run_safe_commands("python -c \"import time; time.sleep(20)\"")
    elapsed = time.time() - start
    
    assert "timed out" in res
    assert elapsed < 18.0


def test_resource_limits_memory(monkeypatch):
    # Temporarily allow the test command
    monkeypatch.setattr("src.tools.coding_tools._is_safe_command", lambda args: True)
    
    res = run_safe_commands("python -c \"import time; x = b'a' * (150 * 1024 * 1024); time.sleep(5)\"")
    assert "memory limit" in res


def test_resource_limits_cpu(monkeypatch):
    # Temporarily allow the test command
    monkeypatch.setattr("src.tools.coding_tools._is_safe_command", lambda args: True)
    
    res = run_safe_commands("python -c \"while True: pass\"")
    assert "CPU time limit" in res

