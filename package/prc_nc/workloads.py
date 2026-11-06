"""Load externally recorded request arrivals without inventing production data.

The GPU runner accepts a trace only when every request has an explicit hard
output cap and an arrival timestamp.  Text is tokenized on the target host;
token-id traces are already immutable inputs.  The returned hashes and
provenance are copied into ``manifest.json`` so a replay cannot silently use a
different trace or tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,120}$")
_TOP_LEVEL_FIELDS = frozenset(("schema_version", "provenance", "requests"))
_PROVENANCE_FIELDS = frozenset(("source", "arrival_kind", "model_id"))
_REQUEST_FIELDS = frozenset(("request_id", "prompt", "prompt_token_ids", "max_tokens", "arrival_s"))
MAX_CONTEXT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 1024


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    """Reject fields that this schema does not define.

    Silently ignoring a misspelled trace field is particularly dangerous here:
    a replay could appear valid while using a different arrival or token cap.
    Schema evolution must therefore introduce a new schema version instead of
    allowing arbitrary extension fields in v1.
    """
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{where} contains unknown fields: {', '.join(unknown)}")


def _nonempty_text(value: Any, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _tokenizer_vocabulary(tokenizer: Any) -> tuple[int | None, frozenset[int] | None]:
    """Return advertised vocabulary information without imposing it artificially.

    Some test and legacy tokenizers intentionally expose no ``vocab_size``.
    Such traces are still accepted after the non-negative integer check.  If a
    tokenizer does advertise a bound, malformed metadata is rejected and token
    IDs are checked against it below.  ``get_vocab`` is preferred because
    Hugging Face added-token IDs can legitimately be above ``vocab_size``.
    """
    if tokenizer is None:
        return None, None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocabulary = get_vocab()
        except Exception as exc:
            raise ValueError("tokenizer get_vocab() failed") from exc
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise ValueError("tokenizer get_vocab() must return a non-empty mapping")
        token_ids = []
        for token, token_id in vocabulary.items():
            if type(token) is not str or type(token_id) is not int or token_id < 0:
                raise ValueError("tokenizer vocabulary must map string tokens to nonnegative integers")
            token_ids.append(token_id)
        return None, frozenset(token_ids)
    if not hasattr(tokenizer, "vocab_size"):
        return None, None
    size = getattr(tokenizer, "vocab_size")
    if callable(size):
        size = size()
    if type(size) is not int or size <= 0:
        raise ValueError("tokenizer vocab_size must be a positive integer when provided")
    return size, None


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is None:
        raise ValueError("text request requires the target tokenizer")
    # Prefer the same chat-template path used by the bundled workloads.  A
    # plain tokenizer callable remains useful for deterministic unit tests and
    # for traces recorded before a chat template was chosen.
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        elif hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            encoded = tokenizer(text)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    except Exception as exc:  # make malformed tokenizer output actionable at the trace boundary
        raise ValueError("tokenizer failed while materializing a text request") from exc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, (list, tuple)):
        raise ValueError("tokenizer must return a one-dimensional token-id sequence")
    return list(ids)


def load_request_trace(
    path: str | Path,
    *,
    count: int | None = None,
    tokenizer: Any = None,
    expected_model_id: str | None = None,
) -> dict:
    """Validate and materialize a single-wave request trace.

    Schema ``prc-request-trace-v1`` is intentionally small and auditable::

        {"schema_version": 1, "provenance": {...},
         "requests": [{"request_id": "...", "prompt_token_ids": [...],
                        "max_tokens": 32, "arrival_s": 0.0}]}

    A request may provide ``prompt`` instead of ``prompt_token_ids``.  No
    request may provide both.  Arrival times must be finite, non-negative and
    non-decreasing; this loader does not resample, shuffle, or infer arrivals.
    ``request_id`` is mandatory in v1 so that every result can be audited back
    to the source row.  The optional ``expected_model_id`` is an exact check
    for callers that know the target artifact; absent that argument, the
    trace's model identifier remains a claim and is not independently attested.
    """
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("count must be a positive integer when provided")
    if expected_model_id is not None:
        _nonempty_text(expected_model_id, "expected_model_id")
    try:
        source = Path(path)
    except TypeError as exc:
        raise ValueError("request trace path must be a filesystem path") from exc
    if not source.is_file():
        raise FileNotFoundError(f"request trace not found: {source}")
    raw_bytes = source.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request trace must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("request trace must be a JSON object")
    _reject_unknown_fields(payload, _TOP_LEVEL_FIELDS, "trace")
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != 1:
        raise ValueError("request trace schema_version must be 1")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("request trace provenance is required")
    _reject_unknown_fields(provenance, _PROVENANCE_FIELDS, "trace provenance")
    for field in ("source", "arrival_kind", "model_id"):
        _nonempty_text(provenance.get(field), f"trace provenance {field}")
    if expected_model_id is not None and provenance["model_id"] != expected_model_id:
        raise ValueError(
            "trace provenance model_id does not match expected_model_id "
            f"({provenance['model_id']!r} != {expected_model_id!r})"
        )
    rows = payload.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("request trace requests must be a non-empty list")
    if count is not None and len(rows) != count:
        raise ValueError(f"trace has {len(rows)} requests, expected configured count {count}")

    vocab_size, vocabulary = _tokenizer_vocabulary(tokenizer)
    prompts: list[list[int]] = []
    caps: list[int] = []
    arrivals: list[float] = []
    selection: list[dict] = []
    seen: set[str] = set()
    previous = -math.inf
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"trace request {index} must be an object")
        _reject_unknown_fields(row, _REQUEST_FIELDS, f"trace request {index}")
        if "request_id" not in row:
            raise ValueError(f"trace request {index} requires request_id")
        request_id = row["request_id"]
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id) or request_id in seen:
            raise ValueError(f"trace request {index} has an invalid or duplicate request_id")
        seen.add(request_id)
        cap = row.get("max_tokens")
        if type(cap) is not int or not 1 <= cap <= MAX_OUTPUT_TOKENS:
            raise ValueError(
                f"trace request {request_id} max_tokens must be an integer in [1, {MAX_OUTPUT_TOKENS}]"
            )
        arrival = row.get("arrival_s")
        if type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < 0 or arrival < previous:
            raise ValueError(f"trace request {request_id} arrival_s must be finite and non-decreasing")
        previous = float(arrival)
        has_ids = "prompt_token_ids" in row
        has_text = "prompt" in row
        if has_ids == has_text:
            raise ValueError(f"trace request {request_id} must contain exactly one prompt representation")
        if has_ids:
            ids = row["prompt_token_ids"]
            if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"trace request {request_id} prompt_token_ids must be non-empty nonnegative integers")
            ids = list(ids)
        else:
            if not isinstance(row["prompt"], str) or not row["prompt"]:
                raise ValueError(f"trace request {request_id} prompt must be a non-empty string")
            ids = _tokenize_text(tokenizer, row["prompt"])
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"tokenizer produced invalid prompt for {request_id}")
        if vocabulary is not None:
            if any(token not in vocabulary for token in ids):
                raise ValueError(f"trace request {request_id} contains a token ID absent from tokenizer vocabulary")
        elif vocab_size is not None and any(token >= vocab_size for token in ids):
            raise ValueError(
                f"trace request {request_id} contains a token ID outside tokenizer vocabulary [0, {vocab_size})"
            )
        if len(ids) + cap > MAX_CONTEXT_TOKENS:
            raise ValueError(
                f"trace request {request_id} exceeds the {MAX_CONTEXT_TOKENS}-token context contract"
            )
        prompts.append(ids)
        caps.append(cap)
        arrivals.append(float(arrival))
        selection.append(
            {
                "request_id": request_id,
                "trace_index": index,
                "prompt_tokens": len(ids),
                "max_tokens": cap,
                "arrival_s": float(arrival),
            }
        )
    return {
        "prompts": prompts,
        "caps": caps,
        "arrivals": arrivals,
        "selection": selection,
        "provenance": dict(provenance),
        "data_sha256": digest,
        "schema_version": 1,
    }


__all__ = ["load_request_trace"]
