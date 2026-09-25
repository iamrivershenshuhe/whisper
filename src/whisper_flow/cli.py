"""Command-line entry point: run the server, or dictate from a file or the mic."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from whisper_flow.config import Config, find_config_file, load_config
from whisper_flow.history import History
from whisper_flow.pipeline import Dictation, DictationResult, EntryNotFound
from whisper_flow.providers import Audio, ProviderError


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(errors="replace")
    load_dotenv()
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    try:
        config = load_config(args.config)
        args.func(args, config)
    except (ProviderError, EntryNotFound, ValueError, OSError) as e:
        raise SystemExit(f"error: {e}") from e


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whisper-flow", description="Voice dictation backend")
    p.add_argument("--config", type=Path, help="path to config.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(required=True, metavar="command")

    s = sub.add_parser("serve", help="run the local HTTP API")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=_serve)

    s = sub.add_parser("dictate", help="transcribe and clean up an audio file")
    s.add_argument("file", type=Path)
    s.add_argument("--app", help="pretend this app is focused (selects the style)")
    s.add_argument("--json", action="store_true", help="print the full result as JSON")
    s.set_defaults(func=_dictate)

    s = sub.add_parser("record", help="record from the microphone, then dictate")
    s.add_argument("--app")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_record)

    s = sub.add_parser("history", help="list recent dictations")
    s.add_argument("-n", "--limit", type=int, default=20)
    s.add_argument("-q", "--query", help="filter by text")
    s.set_defaults(func=_history)

    s = sub.add_parser("retry", help="re-run a stored dictation from its saved audio")
    s.add_argument("id", type=int)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_retry)

    s = sub.add_parser("config", help="show which config file is used and its values")
    s.set_defaults(func=_show_config)
    return p


def _serve(args: argparse.Namespace, config: Config) -> None:
    import uvicorn

    from whisper_flow.server import create_app

    uvicorn.run(
        create_app(config),
        host=args.host or config.server.host,
        port=args.port or config.server.port,
    )


def _dictate(args: argparse.Namespace, config: Config) -> None:
    fmt = args.file.suffix.lstrip(".").lower()
    audio = Audio(args.file.read_bytes(), fmt)
    result = asyncio.run(Dictation.from_config(config).dictate(audio, args.app))
    _print_result(result, args.json)


def _record(args: argparse.Namespace, config: Config) -> None:
    from whisper_flow.audio import record_until_enter

    dictation = Dictation.from_config(config)
    audio = record_until_enter()
    result = asyncio.run(dictation.dictate(audio, args.app))
    _print_result(result, args.json)


def _history(args: argparse.Namespace, config: Config) -> None:
    if not config.history.enabled:
        raise SystemExit("history is disabled in config")
    history = History(config.history.resolved_dir(), save_audio=config.history.save_audio)
    for e in history.list(limit=args.limit, query=args.query):
        text = (e.final_text or e.error or "").replace("\n", " ")
        print(f"#{e.id:<5} {e.created_at:%Y-%m-%d %H:%M}  [{e.status}] {e.app or '-'}  {text}")


def _retry(args: argparse.Namespace, config: Config) -> None:
    result = asyncio.run(Dictation.from_config(config).retry(args.id))
    _print_result(result, args.json)


def _show_config(args: argparse.Namespace, config: Config) -> None:
    path, source = find_config_file(args.config)
    print(f"# source: {source}{f' ({path})' if path else ''}")
    print(f"# data dir: {config.history.resolved_dir()}")
    print(config.model_dump_json(indent=2))


def _print_result(result: DictationResult, as_json: bool) -> None:
    if as_json:
        print(result.model_dump_json(indent=2))
        return
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(result.text)
