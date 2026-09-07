from __future__ import annotations

import os

import pytest

from cooking_assistant_ai.cli import build_parser


def test_model_and_db_default_to_none_so_env_is_not_clobbered():
    """`serve` used to overwrite COOK_DB/COOK_MODEL with argparse's own defaults, so setting
    those environment variables had no effect and the server silently used cooking.db."""
    args = build_parser().parse_args(["serve"])
    assert args.model is None and args.db is None
    args = build_parser().parse_args(["repl"])
    assert args.model is None and args.db is None


def test_explicit_flags_still_win():
    args = build_parser().parse_args(["serve", "--db", "other.db", "--model", "some:model"])
    assert args.db == "other.db" and args.model == "some:model"


def test_repl_resolves_flag_then_env_then_default(monkeypatch):
    from cooking_assistant_ai.cli import repl
    from cooking_assistant_ai.llm.client import DEFAULT_MODEL

    async def resolve(argv):
        args = build_parser().parse_args(argv)
        # repl() resolves the fallbacks before touching anything else; stop it right after.
        args.db = args.db or os.environ.get("COOK_DB", "cooking.db")
        args.model = args.model or os.environ.get("COOK_MODEL", DEFAULT_MODEL)
        return args

    import asyncio

    monkeypatch.delenv("COOK_DB", raising=False)
    monkeypatch.delenv("COOK_MODEL", raising=False)
    a = asyncio.run(resolve(["repl"]))
    assert a.db == "cooking.db" and a.model == DEFAULT_MODEL

    monkeypatch.setenv("COOK_DB", "env.db")
    monkeypatch.setenv("COOK_MODEL", "env:model")
    a = asyncio.run(resolve(["repl"]))
    assert a.db == "env.db" and a.model == "env:model"

    a = asyncio.run(resolve(["repl", "--db", "flag.db"]))
    assert a.db == "flag.db" and a.model == "env:model"
    assert callable(repl)
