"""Tools for the math agent -- Python code execution with sandboxing."""

import builtins
import io
import re
import sys
import threading


class PythonREPLTool:
    """Execute Python code safely with restricted globals and timeout."""

    ALLOWED_MODULES = {
        "math",
        "random",
        "itertools",
        "fractions",
        "decimal",
        "statistics",
        "numpy",
    }

    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "Ellipsis",
        "NotImplemented",
        "__build_class__",
    }

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def _strip_markdown(self, code: str) -> str:
        """Remove markdown code fences from the code string."""
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        return code.strip()

    def _safe_import(self, name, *args, **kwargs):
        """Restrict imports to the allowed module whitelist."""
        if name not in self.ALLOWED_MODULES:
            raise ImportError(
                f"Import of '{name}' is not allowed. Allowed: {self.ALLOWED_MODULES}"
            )
        return builtins.__import__(name, *args, **kwargs)

    def run(self, code: str) -> str:
        """Execute Python code with restricted globals and a timeout."""
        code = self._strip_markdown(code)
        if not code:
            return "Error: No code provided."

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        result_container: dict = {"output": "", "error": None}

        def execute():
            """Inner function to run in a separate thread."""
            try:
                restricted_globals = {
                    "__builtins__": {
                        name: getattr(builtins, name)
                        for name in self.SAFE_BUILTINS
                        if hasattr(builtins, name)
                    } | {"__import__": self._safe_import},
                    "__name__": "__main__",
                }
                exec(code, restricted_globals)
                result_container["output"] = sys.stdout.getvalue()
            except Exception as exc:
                result_container["error"] = f"{type(exc).__name__}: {str(exc)}"

        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)

        sys.stdout = old_stdout

        if thread.is_alive():
            return f"Error: Code execution timed out after {self.timeout} seconds."

        if result_container["error"]:
            return f"Error: {result_container['error']}"

        output = result_container["output"].strip()
        return output if output else "(no output)"
