"""
Names bound by an ``except ... as`` clause must only be used inside that clause.

Python unbinds the exception variable when the handler exits, so a reference to
it further down is an ``UnboundLocalError`` that fires only when that code path
runs. The failure path of a benchmark runner is exactly the code that does not
run in a green test suite, so this is not hypothetical: a coverage change moved
``self._record_failed_queries(..., str(e))`` one dedent too far, outside its
handler, and the whole 178-test suite stayed green while a 3.5-hour Gemma 4 E2B
sweep died on it at the 131072-token group.

An import or a syntax check cannot catch this -- the code parses and compiles
fine. This test reads the AST instead and asserts the property directly, across
the whole package, so it also covers handlers nobody has written yet.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

PACKAGE = Path(__file__).resolve().parents[1] / "crbench"


def python_files() -> Iterator[Path]:
    yield from sorted(PACKAGE.rglob("*.py"))


def _nodes_within(node: ast.AST) -> set[int]:
    """Identity set of every node inside ``node``, itself included."""
    return {id(n) for n in ast.walk(node)}


def check_module(path: Path) -> list[str]:
    """Return a message per out-of-scope use of an exception variable."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []

    handlers = [n for n in ast.walk(tree)
                if isinstance(n, ast.ExceptHandler) and n.name]
    if not handlers:
        return problems

    # For each name bound by any handler, every load of that name must sit
    # inside a handler that binds it.
    bound_names = {h.name for h in handlers}
    scopes: dict[str, list[set[int]]] = {
        name: [_nodes_within(h) for h in handlers if h.name == name]
        for name in bound_names
    }

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        if node.id not in bound_names:
            continue
        if any(id(node) in scope for scope in scopes[node.id]):
            continue
        try:
            where = path.relative_to(PACKAGE.parent)
        except ValueError:          # a file outside the package, e.g. a fixture
            where = path
        problems.append(
            f"{where}:{node.lineno}: "
            f"'{node.id}' is bound by an 'except ... as {node.id}' clause but is "
            f"read outside it; Python unbinds it when the handler exits, so this "
            f"raises UnboundLocalError only when that path is taken"
        )
    return problems


def test_exception_variables_are_not_read_outside_their_handler():
    problems = [p for f in python_files() for p in check_module(f)]
    assert not problems, "\n".join(problems)


def test_the_check_actually_catches_the_bug_it_was_written_for():
    """A guard that cannot fail is worthless; this is the negative control."""
    import tempfile

    broken = (
        "def f(x):\n"
        "    try:\n"
        "        g(x)\n"
        "    except ValueError as e:\n"
        "        log(str(e))\n"
        "    h(str(e))          # <- outside the handler\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "broken.py"
        p.write_text(broken, encoding="utf-8")
        found = check_module(p)
    assert len(found) == 1, found
    assert "UnboundLocalError" in found[0]


def test_reuse_of_the_same_name_in_a_later_handler_is_fine():
    """Two handlers binding 'e' independently must not be flagged."""
    import tempfile

    ok = (
        "def f(x):\n"
        "    try:\n"
        "        g(x)\n"
        "    except ValueError as e:\n"
        "        log(str(e))\n"
        "    try:\n"
        "        g(x)\n"
        "    except KeyError as e:\n"
        "        log(str(e))\n"
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ok.py"
        p.write_text(ok, encoding="utf-8")
        assert check_module(p) == []
