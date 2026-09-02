"""Guards against the phase functions in train.py returning an unbound local.

main() was split into phase functions that hand their results to each other as
tuples. A name assigned only inside a conditional branch still appears in the
return tuple, so a run that skips that branch raises UnboundLocalError before
the first step. That is how the non-resume path broke: ckpt, dataloader_state
and runtime_state were only assigned under `if args.resume is not None`.

No test exercises main(), and reaching one of these takes a real checkpoint on a
GPU, so the check is static.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TRAIN_PY = Path(__file__).resolve().parent.parent / "train.py"
PHASE_PREFIXES = ("_resolve_", "_setup_", "_build_", "_run_")


def _definitely_bound(statements: list[ast.stmt]) -> set[str]:
    """Names bound on every path through `statements`.

    Deliberately conservative: an `if` counts only when it has an `else` and
    both sides bind the name, and loops count for nothing since they may run
    zero times.
    """
    names: set[str] = set()
    for statement in statements:
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
        elif isinstance(statement, ast.If):
            if statement.orelse:
                names |= _definitely_bound(statement.body) & _definitely_bound(statement.orelse)
        elif isinstance(statement, (ast.With, ast.Try)):
            names |= _definitely_bound(statement.body)
    return names


def _phase_functions() -> list[ast.FunctionDef]:
    tree = ast.parse(TRAIN_PY.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith(PHASE_PREFIXES)
    ]


def test_train_py_has_phase_functions() -> None:
    assert {fn.name for fn in _phase_functions()} >= {
        "_resolve_configs",
        "_setup_wandb_and_tokenizers",
        "_build_data",
        "_build_model",
        "_run_training_loop",
    }


@pytest.mark.parametrize("fn", _phase_functions(), ids=lambda fn: fn.name)
def test_returned_names_are_definitely_bound(fn: ast.FunctionDef) -> None:
    final = fn.body[-1]
    if not (isinstance(final, ast.Return) and isinstance(final.value, ast.Tuple)):
        pytest.skip(f"{fn.name} does not end in a tuple return")

    bound = _definitely_bound(fn.body) | {
        arg.arg for arg in (*fn.args.args, *fn.args.kwonlyargs, *fn.args.posonlyargs)
    }
    returned = [e.id for e in final.value.elts if isinstance(e, ast.Name)]
    unbound = [name for name in returned if name not in bound]

    assert not unbound, (
        f"{fn.name} returns {unbound}, which the conservative check cannot prove is "
        f"assigned on every path. Bind them at the top of the function rather than "
        f"only inside a branch."
    )
