import ast
import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger("RAG.CodeParser")

class ASTSymbolVisitor(ast.NodeVisitor):
    """
    AST Visitor to extract classes, functions, methods, and imports.
    """
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.symbols: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []
        self._current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append({
                "type": "import",
                "name": alias.name,
                "asname": alias.asname,
                "line": node.lineno,
                "filepath": self.filepath
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.imports.append({
                "type": "import_from",
                "module": module,
                "name": alias.name,
                "asname": alias.asname,
                "line": node.lineno,
                "filepath": self.filepath
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        prev_class = self._current_class
        self._current_class = node.name
        
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                bases.append(f"{base.value.id}.{base.attr}")
                
        docstring = ast.get_docstring(node) or ""
        
        self.symbols.append({
            "type": "class",
            "name": node.name,
            "filepath": self.filepath,
            "start_line": node.lineno,
            "end_line": node.end_lineno if hasattr(node, "end_lineno") else node.lineno,
            "docstring": docstring,
            "bases": bases,
            "methods": []
        })
        
        self.generic_visit(node)
        self._current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._parse_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._parse_function(node)

    def _parse_function(self, node: Any) -> None:
        docstring = ast.get_docstring(node) or ""
        args = []
        for arg in node.args.args:
            arg_type = ""
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_type = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    arg_type = str(arg.annotation.value)
            args.append({"name": arg.arg, "type": arg_type})

        return_type = ""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                return_type = str(node.returns.value)
                
        symbol_type = "method" if self._current_class else "function"
        
        func_info = {
            "type": symbol_type,
            "name": node.name,
            "filepath": self.filepath,
            "parent_class": self._current_class,
            "start_line": node.lineno,
            "end_line": node.end_lineno if hasattr(node, "end_lineno") else node.lineno,
            "docstring": docstring,
            "arguments": args,
            "return_type": return_type
        }
        
        self.symbols.append(func_info)
        
        # If inside a class, update the class's method list
        if self._current_class:
            for s in self.symbols:
                if s["type"] == "class" and s["name"] == self._current_class:
                    s["methods"].append(node.name)
                    
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = ""
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            call_name = f"{node.func.value.id}.{node.func.attr}"
            
        if call_name:
            self.calls.append({
                "name": call_name,
                "line": node.lineno,
                "filepath": self.filepath
            })
        self.generic_visit(node)


def parse_code_file(filepath: str) -> Dict[str, Any]:
    """
    Parse a python source file using AST and extract structured metadata.
    """
    logger.info(f"Parsing Python source file: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            
        tree = ast.parse(content, filename=filepath)
        visitor = ASTSymbolVisitor(filepath)
        visitor.visit(tree)
        
        return {
            "symbols": visitor.symbols,
            "imports": visitor.imports,
            "calls": visitor.calls,
            "filepath": filepath,
            "lines_count": len(content.splitlines())
        }
    except Exception as e:
        logger.error(f"Error parsing file {filepath} via AST: {e}")
        return {
            "symbols": [],
            "imports": [],
            "calls": [],
            "filepath": filepath,
            "lines_count": 0,
            "error": str(e)
        }
