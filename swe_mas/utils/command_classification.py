"""Shared shell command classifiers for recovery traces.

These helpers are deliberately syntactic and conservative. They are used as the
recovery loop's sensors: if validation or read-only probing is miscounted, the
controller's belief revision and stop decisions become noisy.
"""

from __future__ import annotations

import re


def _segments(command: str) -> list[str]:
    text = str(command or "").replace("\\\n", " ").strip()
    if not text:
        return []
    pieces = re.split(r"(?:&&|;|\n)+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _strip_wrappers(segment: str) -> str:
    text = str(segment or "").strip()
    changed = True
    while changed and text:
        changed = False
        cd_match = re.match(r"^(?:cd|pushd)\s+\S+\s*(?:&&\s*)?(.*)$", text, flags=re.IGNORECASE)
        if cd_match and cd_match.group(1).strip():
            text = cd_match.group(1).strip()
            changed = True
            continue
        timeout_match = re.match(r"^(?:timeout|gtimeout)\s+(?:-k\s+\S+\s+)?\S+\s+(.*)$", text, flags=re.IGNORECASE)
        if timeout_match and timeout_match.group(1).strip():
            text = timeout_match.group(1).strip()
            changed = True
            continue
        env_match = re.match(r"^(?:env\s+)?(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)+(.*)$", text)
        if env_match and env_match.group(1).strip():
            text = env_match.group(1).strip()
            changed = True
    return text


def command_segments(command: str) -> list[str]:
    return [_strip_wrappers(piece) for piece in _segments(command)]


def has_file_redirection(command: str) -> bool:
    if not command:
        return False
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(command):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch == ">":
            nxt = command[idx + 1] if idx + 1 < len(command) else ""
            if nxt == "&":
                continue
            return True
    return False


def looks_like_write_command(command: str) -> bool:
    cmd = str(command or "").strip()
    if not cmd:
        return False
    lowered = cmd.lower()
    if has_file_redirection(cmd):
        return True
    for segment in command_segments(cmd) or [cmd]:
        lowered_segment = segment.lower()
        if re.search(r"\bsed\b.*\s-i(\s|$)", lowered_segment):
            return True
        if re.search(r"\bperl\b.*\s-pi(\s|$)", lowered_segment):
            return True
        if re.search(r"\btee\b(\s|$)", lowered_segment):
            return True
        if re.search(r"\bgit\s+apply\b", lowered_segment):
            return True
        if re.search(r"\b(cat|printf|echo)\b\s+.*\s>\s*\S+", segment):
            return True
        if re.search(r"(^|\s)(python|python3|python\d+(?:\.\d+)?)(\s|$)", lowered_segment):
            python_write_markers = (
                ".write_text(",
                ".write_bytes(",
                ".write(",
                "open(",
            )
            if any(marker in lowered for marker in python_write_markers):
                return True
        for token in (" rm ", " mv ", " cp ", " touch ", " mkdir "):
            if token in f" {lowered_segment} ":
                return True
    return False


def looks_like_validation_command(command: str) -> bool:
    cmd = str(command or "").strip()
    if not cmd:
        return False
    validation_patterns = (
        r"^(?:python|python3|python\d+(?:\.\d+)?)\s+-m\s+(?:pytest|py\.test|unittest|py_compile|django\s+test)\b",
        r"^(?:pytest|py\.test|nosetests|tox)\b",
        r"^(?:(?:python|python3|python\d+(?:\.\d+)?)\s+)?(?:\./|/testbed/)?(?:tests/)?runtests\.py\b",
        r"^(?:(?:python|python3|python\d+(?:\.\d+)?)\s+)?(?:\./|/testbed/)?manage\.py\s+test\b",
        r"^(?:(?:python|python3|python\d+(?:\.\d+)?)\s+)?(?:\./|/testbed/)?setup\.py\s+test\b",
    )
    for segment in command_segments(cmd) or [cmd]:
        lowered = segment.lower()
        if any(re.search(pattern, lowered) for pattern in validation_patterns):
            return True
    return False


def looks_like_readonly_probe_command(command: str) -> bool:
    cmd = str(command or "").strip()
    if not cmd:
        return False
    if looks_like_write_command(cmd) or looks_like_validation_command(cmd):
        return False
    probe_patterns = (
        r"^(ls|pwd|find|grep|rg|cat|head|tail|wc)\b",
        r"^sed\s+-n\b",
    )
    segments = command_segments(cmd) or [cmd]
    meaningful = [segment for segment in segments if not re.match(r"^(?:cd|pushd)\b", segment.lower())]
    if not meaningful:
        meaningful = segments
    return all(any(re.search(pattern, segment.lower()) for pattern in probe_patterns) for segment in meaningful)
