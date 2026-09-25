"""whisper-flow: voice dictation backend (speech-to-text + LLM cleanup)."""

__version__ = "0.1.0"


def main() -> None:
    from whisper_flow.cli import main as cli_main

    cli_main()
