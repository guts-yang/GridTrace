"""Initialize git repo and push to https://github.com/guts-yang/GridTrace."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"d:\Repositories\QUARK-RAG")
REMOTE_URL = "https://github.com/guts-yang/GridTrace.git"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(args)}")
    return subprocess.run(
        list(args),
        cwd=str(REPO),
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def main() -> int:
    # 1) init
    if not (REPO / ".git").exists():
        run("git", "init", "-b", "main")
    else:
        print("(repo already initialized)")

    # 2) remote
    existing = run("git", "remote", "get-url", "origin", check=False)
    if existing.returncode == 0:
        if existing.stdout.strip() != REMOTE_URL:
            run("git", "remote", "set-url", "origin", REMOTE_URL)
    else:
        run("git", "remote", "add", "origin", REMOTE_URL)

    # 3) local user identity (commit-only; do NOT touch global config)
    run("git", "config", "user.name", "GridTrace Bot", check=False)
    run("git", "config", "user.email", "gridtrace@local", check=False)

    # 4) status
    status = run("git", "status", "--short", check=False)
    print(status.stdout)

    # 5) add + commit
    run("git", "add", "-A")
    diff = run("git", "diff", "--cached", "--stat", check=False)
    print(diff.stdout)

    if diff.stdout.strip():
        run(
            "git", "commit", "-m",
            "feat: initial commit of GridTrace (renamed from QUARK)\n\n"
            "- Grid-quantized anchor based RAG with exact unlearning\n"
            "- Two-phase retrieval: L1 anchor routing + L2 rerank\n"
            "- Pluggable storage: pgvector / SQLite / in-memory\n"
            "- Echo / DeepSeek / OpenAI / Ollama generators\n"
            "- FastAPI server + scripts + 61 passing tests",
        )
    else:
        print("(nothing to commit)")

    # 6) push
    push = run("git", "push", "-u", "origin", "main", check=False)
    print("PUSH STDOUT:")
    print(push.stdout)
    print("PUSH STDERR:")
    print(push.stderr)
    print(f"PUSH EXIT: {push.returncode}")
    return push.returncode


if __name__ == "__main__":
    sys.exit(main())
