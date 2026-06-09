from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("RAG.CodeParser")

# Language module mapping
_LANGUAGE_MODULES = {
    "python": ("tree_sitter_python", "language"),
}


def _get_language(language: str):
    """Get the tree-sitter language object for the given language name."""
    if language in _LANGUAGE_MODULES:
        module_name, attr = _LANGUAGE_MODULES[language]
        try:
            mod = __import__(module_name)
            return getattr(mod, attr)()
        except ImportError as e:
            logger.error(f"Tree-sitter {language} language not installed: {e}")
            raise
    raise ValueError(f"Unsupported language: {language}")


def _parse_with_tree_sitter(content: bytes, filepath: str, language: str = "python") -> Dict[str, Any]:
    """Parse source code using Tree-sitter for robust extraction."""
    ts_language = _get_language(language)
    from tree_sitter import Language as TSLanguage, Parser
    
    Language = TSLanguage(ts_language)
    parser = Parser(Language)
    tree = parser.parse(content)
    
    symbols: List[Dict[str, Any]] = []
    imports: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []
    
    _extract_symbols(tree.root_node, symbols, filepath, content)
    _extract_imports(tree.root_node, imports, filepath, content)
    _extract_calls(tree.root_node, calls, filepath, content)
    
    return {
        "symbols": symbols,
        "imports": imports,
        "calls": calls,
        "filepath": filepath,
        "lines_count": content.decode("utf-8", errors="replace").count("\n") + 1,
    }


def _extract_symbols(node, symbols: List[Dict[str, Any]], filepath: str, content: bytes) -> None:
    """Extract class and function definitions from tree-sitter node."""
    
    source = content.decode("utf-8", errors="replace")
    
    if node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        class_name = name_node.text.decode("utf-8") if name_node else "Unknown"
        
        body_node = node.child_by_field_name("body")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        methods = _extract_methods(body_node, source) if body_node else []
        docstring = _extract_docstring(node, source)
        
        bases = _extract_bases(node)
        
        symbols.append({
            "type": "class",
            "name": class_name,
            "filepath": filepath,
            "start_line": start_line,
            "end_line": end_line,
            "docstring": docstring,
            "bases": bases,
            "methods": methods,
        })
        
    elif node.type in ("function_definition", "async_function_definition"):
        name_node = node.child_by_field_name("name")
        func_name = name_node.text.decode("utf-8") if name_node else "Unknown"
        
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        args_node = node.child_by_field_name("parameters")
        arguments = _extract_arguments(args_node) if args_node else []
        
        returns_node = node.child_by_field_name("return_type")
        return_type = _extract_type_annotation(returns_node, content) if returns_node else ""
        
        docstring = _extract_docstring(node, source)
        
        parent_class = _find_parent_class(node)
        symbol_type = "method" if parent_class else "function"
        
        symbols.append({
            "type": symbol_type,
            "name": func_name,
            "filepath": filepath,
            "parent_class": parent_class,
            "start_line": start_line,
            "end_line": end_line,
            "docstring": docstring,
            "arguments": arguments,
            "return_type": return_type,
        })
        
    for child in node.children:
        _extract_symbols(child, symbols, filepath, content)


def _extract_methods(body_node, source: str) -> List[str]:
    """Extract method names from a class body."""
    methods = []
    for child in body_node.children:
        if child.type in ("function_definition", "async_function_definition"):
            name_node = child.child_by_field_name("name")
            if name_node:
                methods.append(name_node.text.decode("utf-8"))
    return methods


def _extract_arguments(args_node) -> List[Dict[str, Any]]:
    """Extract function arguments with type annotations."""
    arguments = []
    
    for child in args_node.children:
        if child.type == "identifier" or child.type == "typed_parameter":
            # Handle regular parameters
            if child.type == "typed_parameter":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")
                name = name_node.text.decode("utf-8") if name_node else ""
                arg_type = type_node.text.decode("utf-8") if type_node else ""
            else:
                name = child.text.decode("utf-8") if child.text else ""
                arg_type = ""
            
            if name:
                arguments.append({"name": name, "type": arg_type})
                
    return arguments


def _extract_type_annotation(type_node, content: bytes) -> str:
    """Extract type annotation from a type node."""
    if type_node and type_node.text:
        return type_node.text.decode("utf-8", errors="replace")
    return ""


def _extract_bases(node) -> List[str]:
    """Extract class inheritance bases."""
    bases = []
    
    for child in node.children:
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type not in ("(", ")", ","):
                    bases.append(arg.text.decode("utf-8", errors="replace"))
    return bases


def _extract_docstring(node, source: str) -> str:
    """Extract docstring from a function/class node."""
    
    # Find the first statement in the body which should be a string for docstring
    body_node = node.child_by_field_name("body")
    if not body_node or not body_node.children:
        return ""
    
    first_stmt = body_node.children[0]
    if first_stmt.type == "expression_statement":
        expr_child = first_stmt.children[0] if first_stmt.children else None
        if expr_child and expr_child.type == "string":
            # Extract string content, removing quotes
            docstring_text = expr_child.text.decode("utf-8", errors="replace")
            # Remove triple quotes or single quotes
            docstring_text = docstring_text.strip('"""').strip("'''").strip('"').strip("'").strip()
            return docstring_text
    
    return ""


def _find_parent_class(node) -> Optional[str]:
    """Find the parent class name by walking up the tree."""
    
    parent = node.parent
    while parent:
        if parent.type == "class_definition":
            name_node = parent.child_by_field_name("name")
            return name_node.text.decode("utf-8") if name_node else None
        parent = parent.parent
    return None


def _extract_imports(node, imports: List[Dict[str, Any]], filepath: str, content: bytes) -> None:
    """Extract import statements from tree-sitter node."""
    
    if node.type == "import_statement":
        import_nodes = []
        for child in node.children:
            if child.type == "dotted_name":
                names = [n.text.decode("utf-8") for n in child.children if n.type not in (",",)]
                if names:
                    import_nodes.append(names)
            elif child.type == "identifier":
                import_nodes.append([child.text.decode("utf-8")])
        
        for name_list in import_nodes:
            module_name = ".".join(name_list)
            imports.append({
                "type": "import",
                "name": module_name,
                "asname": None,
                "line": node.start_point[0] + 1,
                "filepath": filepath,
            })
            
    elif node.type == "import_from_statement":
        dotted_names = []
        for child in node.children:
            if child.type == "dotted_name":
                names = [n.text.decode("utf-8") for n in child.children if n.type not in (",",)]
                dotted_names.append(names[0] if names else "")
        
        module_name = dotted_names[0] if len(dotted_names) > 0 else ""
        
        for name in dotted_names[1:]:
            imports.append({
                "type": "import_from",
                "module": module_name,
                "name": name,
                "asname": None,
                "line": node.start_point[0] + 1,
                "filepath": filepath,
            })
            
    for child in node.children:
        _extract_imports(child, imports, filepath, content)


def _extract_calls(node, calls: List[Dict[str, Any]], filepath: str, content: bytes) -> None:
    """Extract function call expressions from tree-sitter node."""
    
    if node.type == "call":
        func_node = node.child_by_field_name("function")
        
        call_name = ""
        if func_node:
            if func_node.type == "identifier":
                call_name = func_node.text.decode("utf-8")
            elif func_node.type == "attribute":
                # Handle method calls like obj.method()
                obj_node = func_node.child_by_field_name("object")
                attr_node = func_node.child_by_field_name("attribute")
                obj_name = obj_node.text.decode("utf-8") if obj_node else ""
                attr_name = attr_node.text.decode("utf-8") if attr_node else ""
                if obj_name and attr_name:
                    call_name = f"{obj_name}.{attr_name}"
                    
        if call_name:
            calls.append({
                "name": call_name,
                "line": node.start_point[0] + 1,
                "filepath": filepath,
            })
            
    for child in node.children:
        _extract_calls(child, calls, filepath, content)


def parse_code_file(filepath: str, language: str = "python") -> Dict[str, Any]:
    """
    Parse a source file using Tree-sitter and extract structured metadata.
    Returns a dictionary compatible with the original AST parser output format.
    """
    logger.info(f"Parsing {language} source file: {filepath}")
    try:
        with open(filepath, "rb") as f:
            content = f.read()
            
        result = _parse_with_tree_sitter(content, filepath, language)
        return result
    except ImportError as e:
        # Fallback to ast if tree-sitter is not available
        logger.warning(f"Tree-sitter not available, falling back to ast: {e}")
        return _parse_with_ast_fallback(filepath, language)
    except Exception as e:
        logger.error(f"Error parsing file {filepath} via Tree-sitter: {e}")
        return {
            "symbols": [],
            "imports": [],
            "calls": [],
            "filepath": filepath,
            "lines_count": 0,
            "error": str(e),
        }


def _parse_with_ast_fallback(filepath: str, language: str = "python") -> Dict[str, Any]:
    """Fallback to original AST parser if tree-sitter is unavailable. Only supports Python."""
    import ast
    
    if language != "python":
        return {
            "symbols": [],
            "imports": [],
            "calls": [],
            "filepath": filepath,
            "lines_count": 0,
            "error": f"AST fallback only supports Python, got {language}",
        }
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            
        tree = ast.parse(content, filename=filepath)
        
        symbols = []
        imports = []
        calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append({
                    "type": "class",
                    "name": node.name,
                    "filepath": filepath,
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "docstring": ast.get_docstring(node) or "",
                    "bases": [ast.unparse(base) for base in node.bases] if node.bases else [],
                    "methods": [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))],
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [{"name": arg.arg, "type": ""} for arg in node.args.args]
                if node.returns:
                    try:
                        ret_type = ast.unparse(node.returns)
                    except:
                        ret_type = ""
                else:
                    ret_type = ""
                symbols.append({
                    "type": "method" if any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if hasattr(p, "body") and node in getattr(p, "body", [])) else "function",
                    "name": node.name,
                    "filepath": filepath,
                    "parent_class": "",
                    "start_line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "docstring": ast.get_docstring(node) or "",
                    "arguments": args,
                    "return_type": ret_type,
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "name": alias.name,
                        "asname": alias.asname,
                        "line": node.lineno,
                        "filepath": filepath,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append({
                        "type": "import_from",
                        "module": module,
                        "name": alias.name,
                        "asname": alias.asname,
                        "line": node.lineno,
                        "filepath": filepath,
                    })
                    
        return {
            "symbols": symbols,
            "imports": imports,
            "calls": calls,
            "filepath": filepath,
            "lines_count": len(content.splitlines()),
        }
    except Exception as e:
        return {
            "symbols": [],
            "imports": [],
            "calls": [],
            "filepath": filepath,
            "lines_count": 0,
            "error": str(e),
        }