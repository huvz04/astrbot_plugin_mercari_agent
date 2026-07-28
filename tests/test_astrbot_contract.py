from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


_OPTIONAL_ASTRBOT_RUNTIME_MODULES = frozenset({"deprecated"})


def _missing_module_name(stderr: str) -> str | None:
    prefix = "ModuleNotFoundError: No module named "
    missing_line = next(
        (
            line.strip()
            for line in reversed(stderr.splitlines())
            if line.strip().startswith(prefix)
        ),
        None,
    )
    if missing_line is None:
        return None

    try:
        missing_module = ast.literal_eval(missing_line.removeprefix(prefix))
    except (SyntaxError, ValueError):
        return None
    return missing_module if isinstance(missing_module, str) else None


def _assert_real_astrbot_runtime_import(
    result: subprocess.CompletedProcess[str],
) -> None:
    missing_module = _missing_module_name(result.stderr)
    if (
        result.returncode != 0
        and missing_module in _OPTIONAL_ASTRBOT_RUNTIME_MODULES
    ):
        pytest.skip(
            "real AstrBot source export verified, but complete runtime import "
            f"is unavailable: ModuleNotFoundError: No module named {missing_module!r}"
        )
    assert result.returncode == 0, result.stderr


def test_real_astrbot_public_message_chain_contract() -> None:
    """Exercise the adjacent AstrBot source when its optional runtime is available."""
    source_value = os.environ.get("ASTRBOT_SOURCE", "").strip()
    if not source_value:
        pytest.skip("ASTRBOT_SOURCE is not set; real AstrBot API smoke check not requested")

    source = Path(source_value).resolve()
    event_init = source / "astrbot" / "api" / "event" / "__init__.py"
    assert event_init.is_file(), f"ASTRBOT_SOURCE has no public event API: {event_init}"

    tree = ast.parse(event_init.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    exported_names = {
        value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "__all__"
        if isinstance(node.value, (ast.List, ast.Tuple))
        for value in node.value.elts
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    assert "MessageChain" in imported_names
    assert "MessageChain" in exported_names

    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source), existing_path) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from astrbot.api.event import MessageChain; "
                "import astrbot_plugin_mercari_agent.astrbot_adapters; "
                "print(MessageChain.__module__)"
            ),
        ],
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    _assert_real_astrbot_runtime_import(result)


def _failed_runtime_import(missing_module: str) -> subprocess.CompletedProcess[str]:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        f"ModuleNotFoundError: No module named '{missing_module}'\n"
    )
    return subprocess.CompletedProcess(
        args=(sys.executable, "-c", "import plugin"),
        returncode=1,
        stdout="",
        stderr=stderr,
    )


def test_missing_deprecated_runtime_dependency_skips() -> None:
    result = _failed_runtime_import("deprecated")

    with pytest.raises(pytest.skip.Exception, match="deprecated"):
        _assert_real_astrbot_runtime_import(result)


@pytest.mark.parametrize(
    "missing_module",
    [
        "astrbot_plugin_mercari_agent.plugin_dependency",
        "astrbot.api.even",
        "unexpected_dependency",
        "deprecated.extra",
    ],
)
def test_unexpected_missing_runtime_module_fails(missing_module: str) -> None:
    result = _failed_runtime_import(missing_module)

    try:
        _assert_real_astrbot_runtime_import(result)
    except pytest.skip.Exception as exc:
        pytest.fail(f"unexpected skip for {missing_module!r}: {exc}")
    except AssertionError as exc:
        assert (
            f"ModuleNotFoundError: No module named {missing_module!r}" in str(exc)
        )
    else:
        pytest.fail(f"missing module {missing_module!r} did not fail")
