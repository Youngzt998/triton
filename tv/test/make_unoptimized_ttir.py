"""
Generate unoptimized TTIR from a Triton kernel script.

Usage:
    python tv/test/make_ttir.py <script.py> [output_dir]

The output directory defaults to tv/test/TTIR/source/.
The script monkey-patches make_ttir to capture the raw IR before any
optimization passes, then uses triton.compile() to compile only — the
kernel is never launched on the GPU.
"""

import sys
import os
import re
import importlib.util

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <script.py> [output_dir]")
    sys.exit(1)

script_path = os.path.abspath(sys.argv[1])
output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "TTIR", "source")
os.makedirs(output_dir, exist_ok=True)

os.environ["TRITON_ALWAYS_COMPILE"] = "1"

import triton
from triton.backends import backends
from triton.compiler.compiler import ASTSource, compile as triton_compile
from triton.runtime import driver

# Patch make_ttir on all registered backend classes
_patched = []
for _name, _backend in backends.items():
    _cls = _backend.compiler
    _patched.append((_cls, _cls.make_ttir))


def _extract_kernel_name(mod_str):
    m = re.search(r'tt\.func\s+public\s+@(\w+)', mod_str)
    if m:
        return m.group(1)
    m = re.search(r'func\.func\s+(?:public\s+)?@(\w+)', mod_str)
    if m:
        return m.group(1)
    return "unknown"


def _make_dump_wrapper(original_make_ttir):
    """Create a wrapper that captures each backend's own original make_ttir via closure,
    so the correct original is called even when multiple backends are registered."""
    @staticmethod
    def _make_ttir_dump_raw(mod, metadata, opt, capability):
        mod_str = str(mod)
        name = _extract_kernel_name(mod_str)
        out_path = os.path.join(output_dir, f"{name}.ttir")
        with open(out_path, "w") as f:
            f.write(mod_str)
        print(f"Saved unoptimized TTIR -> {out_path}")
        return original_make_ttir(mod, metadata, opt, capability)
    return _make_ttir_dump_raw


for _cls, _orig in _patched:
    _cls.make_ttir = _make_dump_wrapper(_orig)


# Import the user's script as a module (without running top-level code)
# by only loading the kernel definitions decorated with @triton.jit
def _load_kernels(path):
    """Import the script and collect all triton.jit-decorated kernel functions."""
    spec = importlib.util.spec_from_file_location("_user_module", path)
    # Execute only import and kernel definitions by filtering at AST level
    import ast

    with open(path) as f:
        tree = ast.parse(f.read())

    # Keep only: imports, and decorated function defs (the kernels)
    filtered = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            filtered.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            filtered.append(node)
    tree.body = filtered
    ast.fix_missing_locations(tree)

    code = compile(tree, path, "exec")
    namespace = {}
    exec(code, namespace)

    # Collect all triton.jit kernels
    kernels = {}
    for name, obj in namespace.items():
        if isinstance(obj, triton.JITFunction):
            kernels[name] = obj
    return kernels


def _compile_kernel(kernel):
    """Compile a triton.jit kernel without running it, by constructing
    a signature from its argument list and calling triton.compile()."""
    target = driver.active.get_current_target()
    # Build a dummy signature: pointers for ptr args, i32 for scalars
    sig = {}
    constexprs = {}
    for i, param in enumerate(kernel.params):
        if param.is_constexpr:
            # Provide a reasonable default for constexpr block sizes
            constexprs[param.name] = 1024
        else:
            # Heuristic: names ending in _ptr or containing "ptr" are pointers
            if "ptr" in param.name.lower():
                sig[param.name] = "*fp32"
            else:
                sig[param.name] = "i32"

    src = ASTSource(kernel, signature=sig, constexprs=constexprs)
    triton_compile(src, target=target)


kernels = _load_kernels(script_path)
if not kernels:
    print(f"No triton.jit kernels found in {script_path}")
    sys.exit(1)

for name, kernel in kernels.items():
    print(f"Compiling kernel: {name}")
    _compile_kernel(kernel)
