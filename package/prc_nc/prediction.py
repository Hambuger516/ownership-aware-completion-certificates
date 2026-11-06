"""Leakage-safe output-length calibration for the PRC proposal layer.

The predictor is intentionally a small empirical conditional model.  It is
calibrated only from *completed real generation traces*.  Dataset reference
answers are metadata used to identify a prompt, never targets for this model.
The hard output contract remains outside this module: the result here is a
soft proposal (a mean, variance, and an upper reserve), not an admission
decision.

The loader understands the manifests produced by ``gpu_run.py``.  A manifest
must carry the selected ``source_index`` for every request; a request without
that provenance is rejected instead of being assigned an invented identity.
Every progress observation from a trajectory inherits the split of its
``source_index``.  This is the important leakage guard when one request emits
many prefix observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


SPLIT_VERSION = "source-index-sha256-v1"
SPLIT_NAMES = ("calibration", "validation", "test")


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TraceRecord:
    """One progress observation from a completed, real generation.

    ``final_output_tokens`` is available while fitting/evaluating only.  A
    runtime :class:`PredictionQuery` deliberately has no such field.  The
    source identity is the 512-row ``source_index``; ``run_id/request_id``
    identify the independent trajectory inside a real run.
    """

    source_index: int
    run_id: str
    request_id: str
    prompt_tokens: int
    observed_output_tokens: int
    final_output_tokens: int
    category: str = "default"
    terminal: str = "completed"
    source_path: str = ""
    trace_sha256: str = ""
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        _nonnegative_int(self.source_index, "source_index")
        for name in ("prompt_tokens", "observed_output_tokens", "final_output_tokens"):
            _nonnegative_int(getattr(self, name), name)
        if self.final_output_tokens < self.observed_output_tokens:
            raise ValueError("observed output cannot exceed final output")
        for name in ("run_id", "request_id", "category"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.terminal != "completed":
            raise ValueError("only completed real generations may calibrate")

    @property
    def trajectory_id(self) -> str:
        return f"{self.run_id}/{self.request_id}"

    @property
    def remaining_output_tokens(self) -> int:
        return self.final_output_tokens - self.observed_output_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "trajectory_id": self.trajectory_id,
            "remaining_output_tokens": self.remaining_output_tokens,
        }


@dataclass(frozen=True)
class PredictionQuery:
    """Information available before the future output is known."""

    request_id: str
    prompt_tokens: int
    observed_output_tokens: int = 0
    category: str = "default"
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        _nonnegative_int(self.prompt_tokens, "prompt_tokens")
        _nonnegative_int(self.observed_output_tokens, "observed_output_tokens")
        if type(self.category) is not str or not self.category:
            raise ValueError("category must be a non-empty string")
        if self.max_output_tokens is not None:
            _nonnegative_int(self.max_output_tokens, "max_output_tokens")


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    remaining_mean_tokens: float
    remaining_variance_tokens2: float
    soft_upper_tokens: int
    calibration_state: str
    calibration_version: str
    support: str
    support_count: int

    def __post_init__(self) -> None:
        for name in ("remaining_mean_tokens", "remaining_variance_tokens2"):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        _nonnegative_int(self.soft_upper_tokens, "soft_upper_tokens")
        _nonnegative_int(self.support_count, "support_count")
        if self.calibration_state not in {"calibrated", "unfitted"}:
            raise ValueError("unknown calibration state")


@dataclass(frozen=True)
class SplitManifest:
    """Deterministic source-index partition and provenance summary."""

    version: str
    seed: int
    ratios: tuple[float, float, float]
    source_indices: tuple[int, ...]
    assignments: Mapping[int, str]

    def __post_init__(self) -> None:
        if self.version != SPLIT_VERSION:
            raise ValueError("unsupported split manifest version")
        if len(self.ratios) != 3 or any(x < 0 for x in self.ratios):
            raise ValueError("ratios must contain three non-negative values")
        if not self.source_indices:
            raise ValueError("source index set cannot be empty")
        if set(self.source_indices) != set(self.assignments):
            raise ValueError("split assignments must cover every source index exactly")
        if any(value not in SPLIT_NAMES for value in self.assignments.values()):
            raise ValueError("unknown split name")

    @property
    def counts(self) -> dict[str, int]:
        return {name: sum(self.assignments[i] == name for i in self.source_indices) for name in SPLIT_NAMES}

    @property
    def sha256(self) -> str:
        payload = {
            "version": self.version,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "source_indices": list(self.source_indices),
            "assignments": {str(k): self.assignments[k] for k in self.source_indices},
        }
        return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "source_indices": list(self.source_indices),
            "assignments": {str(k): self.assignments[k] for k in self.source_indices},
            "counts": self.counts,
            "sha256": self.sha256,
        }

    def split_of(self, source_index: int) -> str:
        try:
            return self.assignments[source_index]
        except KeyError as exc:
            raise KeyError(f"source_index {source_index} is absent from split manifest") from exc


def make_split_manifest(
    source_indices: Iterable[int],
    *,
    seed: int = 20260919,
    ratios: tuple[float, float, float] = (0.60, 0.20, 0.20),
) -> SplitManifest:
    """Create a deterministic 60/20/20 source-identity split.

    Assignment is made after a stable hash sort, never by input order.  The
    first two partitions use floor counts and the remainder is assigned to the
    test partition, yielding 307/102/103 for the required 512 rows.
    """

    ids = sorted({_nonnegative_int(x, "source_index") for x in source_indices})
    if not ids:
        raise ValueError("at least one source index is required")
    if len(ratios) != 3 or any(not math.isfinite(float(x)) or float(x) < 0 for x in ratios):
        raise ValueError("ratios must be finite and non-negative")
    total_ratio = sum(float(x) for x in ratios)
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    ordered = sorted(ids, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode("ascii")).hexdigest())
    n = len(ordered)
    n_cal = math.floor(n * float(ratios[0]))
    n_val = math.floor(n * float(ratios[1]))
    assignments = {
        source: ("calibration" if pos < n_cal else "validation" if pos < n_cal + n_val else "test")
        for pos, source in enumerate(ordered)
    }
    return SplitManifest(SPLIT_VERSION, int(seed), tuple(float(x) for x in ratios), tuple(ids), assignments)


@dataclass(frozen=True)
class TraceCorpus:
    records: tuple[TraceRecord, ...]
    split_manifest: SplitManifest
    rejected: tuple[Mapping[str, Any], ...] = ()
    source_rows_sha256: str = ""

    def records_for(self, split: str) -> tuple[TraceRecord, ...]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}")
        return tuple(x for x in self.records if self.split_manifest.split_of(x.source_index) == split)

    @property
    def trajectory_splits(self) -> dict[str, str]:
        return {x.trajectory_id: self.split_manifest.split_of(x.source_index) for x in self.records}


def load_source_rows(path: str | Path, *, expected_count: int = 512) -> tuple[dict[str, Any], ...]:
    """Load prompt metadata without treating reference answers as labels."""

    path = Path(path)
    raw = path.read_bytes()
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("source data must be a JSON list")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} source rows, found {len(rows)}")
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or "source_index" not in row:
            raise ValueError("every source row needs source_index")
        idx = _nonnegative_int(row["source_index"], "source_index")
        if idx in seen:
            raise ValueError(f"duplicate source_index {idx}")
        seen.add(idx)
        # Keep only features available before generation.  In particular,
        # reference_response is intentionally omitted from this representation.
        out.append({"source_index": idx, "category": str(row.get("category", "default"))})
    return tuple(out)


def _progress_points(final_tokens: int) -> range:
    # Include prompt-time observation and each generated prefix.  The final
    # point is useful for calibration sanity checks and remains in the same
    # source split as all earlier points.
    return range(final_tokens + 1)


def load_real_generation_traces(
    root: str | Path,
    *,
    source_rows: Sequence[Mapping[str, Any]] | None = None,
    split_manifest: SplitManifest | None = None,
    expected_source_count: int | None = 512,
    strict: bool = True,
) -> TraceCorpus:
    """Collect completed traces and progress observations from real run dirs.

    Each run directory must contain ``manifest.json`` and ``requests.json``.
    ``manifest.selection[index].source_index`` is the only accepted mapping
    from a generated request to the 512-row source identity.  Requests without
    a completed terminal state, token list, or selection provenance are
    reported in ``rejected`` and never enter calibration.
    """

    root = Path(root)
    if source_rows is None:
        default = root / "dolly_pilot_sample.json"
        if default.exists():
            source_rows = load_source_rows(default, expected_count=expected_source_count)
        else:
            source_rows = ()
    row_by_index = {int(x["source_index"]): x for x in source_rows}
    if expected_source_count is not None and source_rows and len(row_by_index) != expected_source_count:
        raise ValueError("source row count does not match expected count")
    if split_manifest is None:
        if not row_by_index:
            raise ValueError("source rows are required to create a split manifest")
        split_manifest = make_split_manifest(row_by_index)
    if set(row_by_index) and set(row_by_index) != set(split_manifest.source_indices):
        raise ValueError("source rows and split manifest identities differ")

    records: list[TraceRecord] = []
    rejected: list[Mapping[str, Any]] = []
    run_dirs = sorted({p.parent for p in root.rglob("requests.json") if (p.parent / "manifest.json").exists()})
    for run_dir in run_dirs:
        requests_path = run_dir / "requests.json"
        manifest_path = run_dir / "manifest.json"
        # A run is an atomic provenance unit.  Build its observations in a
        # private list and publish them only after every request in the run
        # has passed validation.  This prevents a malformed late request from
        # leaving an apparently valid prefix of that run in calibration data.
        run_records: list[TraceRecord] = []
        try:
            requests = json.loads(requests_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selection = manifest.get("selection")
            if not isinstance(selection, list):
                raise ValueError("manifest.selection is missing")
            run_id = str(manifest.get("configuration", {}).get("run_id") or run_dir.name)
            trace_hash = _sha256_bytes(requests_path.read_bytes())
            manifest_hash = _sha256_bytes(manifest_path.read_bytes())
            for req in requests:
                idx = req.get("index")
                if type(idx) is not int or idx < 0 or idx >= len(selection):
                    raise ValueError(f"request index lacks selection provenance: {idx!r}")
                selected = selection[idx]
                source_index = selected.get("source_index") if isinstance(selected, Mapping) else None
                if type(source_index) is not int or source_index not in split_manifest.assignments:
                    raise ValueError(f"request {req.get('request_id')} lacks source_index provenance")
                if source_rows and source_index not in row_by_index:
                    raise ValueError(f"source_index {source_index} absent from source rows")
                terminal = str(req.get("terminal", ""))
                tokens = req.get("tokens")
                if terminal != "completed" or not isinstance(tokens, list) or not tokens:
                    raise ValueError(f"request {req.get('request_id')} is not a completed token trace")
                final = len(tokens)
                prompt_tokens = selected.get("prompt_tokens")
                if type(prompt_tokens) is not int or prompt_tokens < 0:
                    raise ValueError(f"request {req.get('request_id')} lacks prompt token provenance")
                category = str(row_by_index.get(source_index, {}).get("category", "default"))
                request_id = str(req.get("request_id"))
                for observed in _progress_points(final):
                    run_records.append(TraceRecord(
                        source_index=source_index,
                        run_id=run_id,
                        request_id=request_id,
                        prompt_tokens=prompt_tokens,
                        observed_output_tokens=observed,
                        final_output_tokens=final,
                        category=category,
                        source_path=str(requests_path),
                        trace_sha256=trace_hash,
                        manifest_sha256=manifest_hash,
                    ))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            rejected.append({"run_path": str(run_dir), "reason": str(exc)})
            if strict:
                # A strict calibration request must not silently turn a
                # malformed run into a partial corpus.  The CLI explicitly
                # chooses strict=False when it wants a documented partial
                # collection and records the rejected runs in its sidecar.
                raise ValueError(f"malformed real generation run {run_dir}: {exc}") from exc
        else:
            records.extend(run_records)
    if not records and strict:
        raise ValueError("no completed real generation traces with provenance were found")
    return TraceCorpus(tuple(records), split_manifest, tuple(rejected), "")


def _prompt_bucket(prompt_tokens: int) -> int:
    return min(32, max(0, prompt_tokens // 32))


def _progress_bucket(observed_tokens: int) -> int:
    # The bucket uses only observed progress at query time, never the future
    # final length.
    return min(10, int(math.log2(observed_tokens + 1)))


@dataclass
class _Moment:
    values: list[float] = field(default_factory=list)
    _summary: tuple[int, float, float] | None = None

    def add(self, value: float) -> None:
        self._summary = None
        self.values.append(float(value))

    @property
    def n(self) -> int:
        if self._summary is not None:
            return self._summary[0]
        return len(self.values)

    @property
    def mean(self) -> float:
        if self._summary is not None:
            return self._summary[1]
        return fmean(self.values) if self.values else 0.0

    @property
    def variance(self) -> float:
        if self._summary is not None:
            return self._summary[2]
        if len(self.values) < 2:
            return 0.0
        mean = self.mean
        return sum((x - mean) ** 2 for x in self.values) / len(self.values)


class ConditionalResidualPredictor:
    """Empirical conditional mean/variance predictor with a hard-cap clip."""

    FORMAT = "prc-real-conditional-predictor-v1"

    def __init__(self, *, z_value: float = 2.0, calibration_version: str = "real-trace-v1") -> None:
        self.z_value = _finite(z_value, "z_value")
        if self.z_value < 0:
            raise ValueError("z_value must be non-negative")
        if not calibration_version:
            raise ValueError("calibration_version must be non-empty")
        self.calibration_version = str(calibration_version)
        self._groups: dict[tuple[Any, ...], _Moment] = {}
        self._fitted = False
        self._fit_source_indices: tuple[int, ...] = ()
        self._fit_split_sha256 = ""
        self._fit_provenance: tuple[str, ...] = ()
        self._fit_manifest_provenance: tuple[str, ...] = ()
        self._metrics: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def calibration_state(self) -> str:
        return "calibrated" if self._fitted else "unfitted"

    @property
    def fit_split_sha256(self) -> str:
        """Hash of the frozen source-identity split used for calibration."""
        return self._fit_split_sha256

    @property
    def fit_trace_sha256(self) -> tuple[str, ...]:
        """Hashes of completed real traces used for calibration."""
        return self._fit_provenance

    @property
    def fit_manifest_sha256(self) -> tuple[str, ...]:
        """Hashes of run manifests carrying request-to-source provenance."""
        return self._fit_manifest_provenance

    @staticmethod
    def _coerce(record: TraceRecord | Mapping[str, Any]) -> TraceRecord:
        if isinstance(record, TraceRecord):
            return record
        if not isinstance(record, Mapping):
            raise TypeError("records must be TraceRecord or mappings")
        required = ("source_index", "run_id", "request_id", "prompt_tokens", "observed_output_tokens", "final_output_tokens")
        if any(key not in record for key in required):
            raise ValueError("record lacks real-trace provenance fields")
        return TraceRecord(
            source_index=int(record["source_index"]), run_id=str(record["run_id"]),
            request_id=str(record["request_id"]), prompt_tokens=int(record["prompt_tokens"]),
            observed_output_tokens=int(record["observed_output_tokens"]),
            final_output_tokens=int(record["final_output_tokens"]), category=str(record.get("category", "default")),
            terminal=str(record.get("terminal", "completed")), source_path=str(record.get("source_path", "")),
            trace_sha256=str(record.get("trace_sha256", "")), manifest_sha256=str(record.get("manifest_sha256", "")),
        )

    @staticmethod
    def _keys(category: str, prompt_tokens: int, observed_tokens: int) -> tuple[tuple[Any, ...], ...]:
        return (
            ("exact", category, _prompt_bucket(prompt_tokens), _progress_bucket(observed_tokens)),
            ("category_progress", category, _progress_bucket(observed_tokens)),
            ("category", category),
            ("global_progress", _progress_bucket(observed_tokens)),
            ("global",),
        )

    def _lookup(self, query: PredictionQuery) -> tuple[_Moment, str]:
        for key in self._keys(query.category, query.prompt_tokens, query.observed_output_tokens):
            moment = self._groups.get(key)
            if moment and moment.n:
                return moment, ":".join(str(x) for x in key)
        raise RuntimeError("calibration has no support for this query")

    def fit(
        self,
        records: Iterable[TraceRecord | Mapping[str, Any]],
        *,
        split_manifest: SplitManifest | None = None,
        calibration_split: str = "calibration",
        validation_records: Iterable[TraceRecord | Mapping[str, Any]] | None = None,
    ) -> "ConditionalResidualPredictor":
        if calibration_split != "calibration":
            raise ValueError("the calibration split is fixed to 'calibration'")
        rows = [self._coerce(x) for x in records]
        if not rows:
            raise ValueError("at least one real generated progress record is required")
        if split_manifest is None:
            split_manifest = make_split_manifest({x.source_index for x in rows})
        self._groups = {}
        fit_ids: set[int] = set()
        provenance: set[str] = set()
        manifest_provenance: set[str] = set()
        for row in rows:
            if split_manifest.split_of(row.source_index) != calibration_split:
                raise ValueError("fit received a validation/test source identity")
            if not row.trace_sha256 or not row.manifest_sha256:
                raise ValueError("real trace provenance hashes are required")
            fit_ids.add(row.source_index)
            provenance.add(row.trace_sha256)
            manifest_provenance.add(row.manifest_sha256)
            for key in self._keys(row.category, row.prompt_tokens, row.observed_output_tokens):
                self._groups.setdefault(key, _Moment()).add(row.remaining_output_tokens)
        self._fitted = True
        self._fit_source_indices = tuple(sorted(fit_ids))
        self._fit_split_sha256 = split_manifest.sha256
        self._fit_provenance = tuple(sorted(provenance))
        self._fit_manifest_provenance = tuple(sorted(manifest_provenance))
        if validation_records is not None:
            self._metrics = self.evaluate(validation_records, split_manifest=split_manifest, split="validation")
        return self

    def predict(self, query: PredictionQuery) -> PredictionResult:
        if not isinstance(query, PredictionQuery):
            raise TypeError("predict expects PredictionQuery")
        if not self._fitted:
            raise RuntimeError("predictor is not calibrated from real traces")
        moment, support = self._lookup(query)
        mean = max(0.0, moment.mean)
        variance = max(0.0, moment.variance)
        reserve = max(0, math.ceil(mean + self.z_value * math.sqrt(variance)))
        if query.max_output_tokens is not None:
            residual_cap = max(0, query.max_output_tokens - query.observed_output_tokens)
            mean = min(mean, residual_cap)
            reserve = min(reserve, residual_cap)
        return PredictionResult(query.request_id, mean, variance, reserve, self.calibration_state,
                                self.calibration_version, support, moment.n)

    def evaluate(
        self,
        records: Iterable[TraceRecord | Mapping[str, Any]],
        *,
        split_manifest: SplitManifest | None = None,
        split: str = "validation",
    ) -> dict[str, Any]:
        rows = [self._coerce(x) for x in records]
        if split_manifest is not None:
            rows = [x for x in rows if split_manifest.split_of(x.source_index) == split]
        if not rows:
            return {"split": split, "n": 0, "mae": None, "rmse": None, "coverage": None}
        errors: list[float] = []
        covered = 0
        for row in rows:
            result = self.predict(PredictionQuery(row.trajectory_id, row.prompt_tokens, row.observed_output_tokens, row.category))
            actual = float(row.remaining_output_tokens)
            errors.append(result.remaining_mean_tokens - actual)
            covered += int(actual <= result.soft_upper_tokens)
        return {
            "split": split,
            "n": len(rows),
            "mae": fmean(abs(x) for x in errors),
            "rmse": math.sqrt(fmean(x * x for x in errors)),
            "coverage": covered / len(rows),
        }

    def checkpoint_dict(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("cannot checkpoint an unfitted predictor")
        groups = {
            json.dumps(key, separators=(",", ":")): {
                "n": value.n, "mean": value.mean, "variance": value.variance,
            }
            for key, value in sorted(self._groups.items(), key=lambda item: repr(item[0]))
        }
        return {
            "format": self.FORMAT,
            "z_value": self.z_value,
            "calibration_version": self.calibration_version,
            "fit_source_indices": list(self._fit_source_indices),
            "fit_split_sha256": self._fit_split_sha256,
            "fit_trace_sha256": list(self._fit_provenance),
            "fit_manifest_sha256": list(self._fit_manifest_provenance),
            "groups": groups,
            "validation_metrics": self._metrics,
        }

    def save_checkpoint(self, path: str | Path) -> str:
        payload = json.dumps(self.checkpoint_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return _sha256_bytes(payload)

    def load_checkpoint(self, path: str | Path, *, expected_split_sha256: str | None = None) -> str:
        path = Path(path)
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        required = {
            "format", "z_value", "calibration_version", "fit_source_indices",
            "fit_split_sha256", "fit_trace_sha256", "fit_manifest_sha256",
            "groups", "validation_metrics",
        }
        if set(payload) != required or payload.get("format") != self.FORMAT:
            raise ValueError("predictor checkpoint format mismatch")
        if expected_split_sha256 is not None and payload.get("fit_split_sha256") != expected_split_sha256:
            raise ValueError("predictor checkpoint split manifest mismatch")
        if not payload.get("fit_trace_sha256") or not payload.get("fit_manifest_sha256") or not payload.get("fit_source_indices"):
            raise ValueError("predictor checkpoint lacks real-trace provenance")
        z_value = _finite(payload["z_value"], "z_value")
        if z_value < 0:
            raise ValueError("checkpoint z_value must be non-negative")
        calibration_version = payload["calibration_version"]
        if type(calibration_version) is not str or not calibration_version:
            raise ValueError("checkpoint calibration_version is malformed")
        source_indices = payload["fit_source_indices"]
        if (type(source_indices) is not list or not source_indices
                or any(type(x) is not int or x < 0 for x in source_indices)
                or source_indices != sorted(set(source_indices))):
            raise ValueError("checkpoint fit_source_indices are malformed")
        trace_hashes = payload["fit_trace_sha256"]
        manifest_hashes = payload["fit_manifest_sha256"]
        if (type(trace_hashes) is not list or not trace_hashes or any(type(x) is not str or not x for x in trace_hashes)
                or type(manifest_hashes) is not list or not manifest_hashes
                or any(type(x) is not str or not x for x in manifest_hashes)):
            raise ValueError("checkpoint provenance hashes are malformed")
        groups_payload = payload["groups"]
        if type(groups_payload) is not dict or not groups_payload:
            raise ValueError("checkpoint groups are malformed")
        groups: dict[tuple[Any, ...], _Moment] = {}
        for encoded, value in groups_payload.items():
            if type(encoded) is not str or type(value) is not dict or set(value) != {"n", "mean", "variance"}:
                raise ValueError("checkpoint group schema mismatch")
            key = tuple(json.loads(encoded))
            # The checkpoint stores sufficient statistics rather than the raw
            # trajectories.  Keep those statistics exact; reconstructing a
            # synthetic list changes the population variance when n > 2.
            n = value["n"]
            mean = _finite(value["mean"], "checkpoint group mean")
            variance = _finite(value["variance"], "checkpoint group variance")
            if type(n) is not int or n <= 0 or variance < 0:
                raise ValueError("checkpoint group moments are malformed")
            moment = _Moment()
            moment._summary = (n, mean, variance)
            groups[key] = moment
        metrics = payload["validation_metrics"]
        if type(metrics) is not dict:
            raise ValueError("checkpoint validation_metrics are malformed")
        # Commit only after the complete payload has passed validation.
        self.z_value = z_value
        self.calibration_version = calibration_version
        self._groups = groups
        self._fitted = True
        self._fit_source_indices = tuple(source_indices)
        self._fit_split_sha256 = str(payload["fit_split_sha256"])
        self._fit_provenance = tuple(trace_hashes)
        self._fit_manifest_provenance = tuple(manifest_hashes)
        self._metrics = dict(metrics)
        return _sha256_bytes(raw)


def load_runtime_predictor(path, *, z_value):
    """Keep fitted moments/provenance but apply the explicit runtime reserve knob."""
    predictor = ConditionalResidualPredictor(z_value=z_value)
    requested_z = predictor.z_value  # Validate before loading the checkpoint.
    digest = predictor.load_checkpoint(path)
    predictor.z_value = requested_z
    return predictor, digest


# Compatibility names used by the existing proposal code and by the runtime
# integration.  They intentionally point to the strict real-trace class.
ResidualLengthPredictor = ConditionalResidualPredictor
LengthPrediction = PredictionResult


__all__ = [
    "TraceRecord", "PredictionQuery", "PredictionResult", "LengthPrediction",
    "SplitManifest", "TraceCorpus", "make_split_manifest", "load_source_rows",
    "load_real_generation_traces", "ConditionalResidualPredictor", "ResidualLengthPredictor",
]
