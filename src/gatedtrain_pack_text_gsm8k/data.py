"""Convert GSM8K rows (`{"question", "answer"}`, the answer ending with a `#### <number>` line) to the intake
format (`{"input", "label"}`). Refusals name line numbers and reasons only, never row content: the data may be sealed
later."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from gatedtrain.errors import Problem, Refused

from .answer import normalise

MARKER = "####"


def _label(answer: str) -> str | None:
    last = answer.split("\n")[-1].strip()
    if not last.startswith(MARKER):
        return None
    return normalise(last[len(MARKER):])


def _parse(line: str) -> tuple[str, str] | str:
    """(question, label) or the reason the line is refused."""
    try:
        row = json.loads(line)
    except ValueError:
        return "not valid JSON"
    if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
        return "not an object with string 'question' and 'answer'"
    label = _label(row["answer"])
    if label is None:
        return "the answer does not end with a '#### <number>' line"
    return row["question"], label


def convert(src: Path, dst: Path) -> dict[str, Any]:
    """Write `dst` and return `{"rows", "duplicates", "refused"}`. When any row is refused nothing is written; an
    empty source is refused (TXT-1) and nothing is written."""
    dst = Path(dst)
    if os.path.lexists(dst):
        raise Refused([Problem("TXT-1", "dst", "the destination file already exists", "choose a new --dst")])
    try:
        text = Path(src).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Refused([Problem("TXT-1", "src", f"cannot read the source as UTF-8 ({type(exc).__name__})",
                               "pass a UTF-8 JSONL file")]) from None
    lines = text.split("\n")  # never splitlines(): a question may hold U+2028
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise Refused([Problem("TXT-1", "src", "the source file holds no row (it is empty)",
                               "pass the GSM8K JSONL file, one {question, answer} object per line")])
    labels: dict[str, str] = {}
    kept: list[tuple[str, str]] = []
    refused: list[dict[str, Any]] = []
    duplicates = 0
    for no, line in enumerate(lines, 1):
        got = _parse(line)
        if isinstance(got, str):
            refused.append({"line": no, "reason": got})
            continue
        question, label = got
        if question in labels:
            if labels[question] != label:
                refused.append({"line": no, "reason": "the same question has a different label on an earlier line"})
            else:
                duplicates += 1
            continue
        labels[question] = label
        kept.append((question, label))
    if refused:
        return {"rows": 0, "duplicates": duplicates, "refused": refused}
    data = "".join(json.dumps({"input": q, "label": lab}, ensure_ascii=False) + "\n" for q, lab in kept).encode("utf-8")
    try:
        fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    except FileExistsError:
        raise Refused([Problem("TXT-1", "dst", "the destination file already exists", "choose a new --dst")]) from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(dst, 0o644)
    return {"rows": len(kept), "duplicates": duplicates, "refused": []}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gatedtrain-pack-text-gsm8k-data", description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="GSM8K JSONL (question, answer)")
    ap.add_argument("--dst", required=True, type=Path, help="intake JSONL to create (must not exist)")
    args = ap.parse_args(argv)
    try:
        summary = convert(args.src, args.dst)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(json.dumps(summary))
    return 3 if summary["refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
