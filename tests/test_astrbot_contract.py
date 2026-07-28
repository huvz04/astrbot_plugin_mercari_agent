from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


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
    if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
        missing_line = next(
            (
                line.strip()
                for line in reversed(result.stderr.splitlines())
                if "ModuleNotFoundError" in line
            ),
            "optional AstrBot runtime dependency is unavailable",
        )
        pytest.skip(
            "real AstrBot source export verified, but complete runtime import "
            f"is unavailable: {missing_line}"
        )
    assert result.returncode == 0, result.stderr
