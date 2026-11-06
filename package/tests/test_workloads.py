import json
from pathlib import Path
import tempfile
import unittest

from prc_nc.configuration import validate_config
from prc_nc.workloads import load_request_trace


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 for ch in text]


class _VocabTokenizer(_Tokenizer):
    vocab_size = 4


class _AddedTokenTokenizer(_Tokenizer):
    vocab_size = 4

    def get_vocab(self):
        # Added special tokens may be assigned IDs above vocab_size.
        return {"a": 0, "b": 1, "<added>": 7}


def _write(root: Path, payload: dict) -> Path:
    path = root / "trace.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class WorkloadTraceTests(unittest.TestCase):
    def test_malformed_path_is_reported_as_value_error(self):
        with self.assertRaises(ValueError):
            load_request_trace(None)

    def test_token_id_trace_is_hashable_and_order_preserving(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _write(
                Path(temp),
                {
                    "schema_version": 1,
                    "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                    "requests": [
                        {"request_id": "r0", "prompt_token_ids": [1, 2], "max_tokens": 3, "arrival_s": 0},
                        {"request_id": "r1", "prompt_token_ids": [4], "max_tokens": 4, "arrival_s": 0.25},
                    ],
                },
            )
            result = load_request_trace(path, count=2)
            self.assertEqual(result["caps"], [3, 4])
            self.assertEqual(result["arrivals"], [0.0, 0.25])
            self.assertEqual([r["request_id"] for r in result["selection"]], ["r0", "r1"])
            self.assertEqual(len(result["data_sha256"]), 64)

    def test_text_trace_uses_target_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _write(
                Path(temp),
                {
                    "schema_version": 1,
                    "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                    "requests": [{"request_id": "r0", "prompt": "abc", "max_tokens": 2, "arrival_s": 0}],
                },
            )
            self.assertEqual(load_request_trace(path, tokenizer=_Tokenizer())["prompts"], [[0, 1, 2]])

    def test_tokenizer_vocabulary_is_checked_when_advertised(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _write(
                Path(temp),
                {
                    "schema_version": 1,
                    "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                    "requests": [
                        {"request_id": "r0", "prompt_token_ids": [3], "max_tokens": 1, "arrival_s": 0},
                    ],
                },
            )
            self.assertEqual(load_request_trace(path, tokenizer=_VocabTokenizer())["prompts"], [[3]])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["requests"][0]["prompt_token_ids"] = [4]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_request_trace(path, tokenizer=_VocabTokenizer())

    def test_added_token_ids_use_get_vocab_not_scalar_vocab_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _write(
                Path(temp),
                {
                    "schema_version": 1,
                    "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                    "requests": [
                        {"request_id": "r0", "prompt_token_ids": [7], "max_tokens": 1, "arrival_s": 0},
                    ],
                },
            )
            self.assertEqual(load_request_trace(path, tokenizer=_AddedTokenTokenizer())["prompts"], [[7]])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["requests"][0]["prompt_token_ids"] = [6]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_request_trace(path, tokenizer=_AddedTokenTokenizer())

    def test_exact_model_provenance_check_is_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            path = _write(
                Path(temp),
                {
                    "schema_version": 1,
                    "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                    "requests": [
                        {"request_id": "r0", "prompt_token_ids": [1], "max_tokens": 1, "arrival_s": 0},
                    ],
                },
            )
            self.assertEqual(load_request_trace(path, expected_model_id="toy")["provenance"]["model_id"], "toy")
            with self.assertRaises(ValueError):
                load_request_trace(path, expected_model_id="different-model")

    def test_invalid_order_and_missing_provenance_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = {
                "schema_version": 1,
                "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                "requests": [
                    {"request_id": "r0", "prompt_token_ids": [1], "max_tokens": 1, "arrival_s": 1},
                    {"request_id": "r1", "prompt_token_ids": [1], "max_tokens": 1, "arrival_s": 0},
                ],
            }
            with self.assertRaises(ValueError):
                load_request_trace(_write(root, bad))
            bad["provenance"] = {}
            with self.assertRaises(ValueError):
                load_request_trace(_write(root, bad))

    def test_v1_rejects_unknown_fields_and_boolean_schema_or_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = {
                "schema_version": 1,
                "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                "requests": [
                    {"request_id": "r0", "prompt_token_ids": [1], "max_tokens": 1, "arrival_s": 0},
                ],
            }
            for location, key in ((base, "extra"), (base["provenance"], "extra"), (base["requests"][0], "extra")):
                mutated = json.loads(json.dumps(base))
                target = mutated if location is base else (
                    mutated["provenance"] if location is base["provenance"] else mutated["requests"][0]
                )
                target[key] = "must-not-be-ignored"
                with self.subTest(location=key):
                    with self.assertRaises(ValueError):
                        load_request_trace(_write(root, mutated))
            mutated = json.loads(json.dumps(base))
            mutated["schema_version"] = True
            with self.assertRaises(ValueError):
                load_request_trace(_write(root, mutated))
            with self.assertRaises(ValueError):
                load_request_trace(_write(root, base), count=True)

    def test_request_id_is_explicit_and_context_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = {
                "schema_version": 1,
                "provenance": {"source": "unit", "arrival_kind": "recorded", "model_id": "toy"},
                "requests": [{"prompt_token_ids": [1], "max_tokens": 1, "arrival_s": 0}],
            }
            with self.assertRaises(ValueError):
                load_request_trace(_write(root, base))
            base["requests"][0] = {
                "request_id": "r0",
                "prompt_token_ids": [1] * 1024,
                "max_tokens": 1,
                "arrival_s": 0,
            }
            with self.assertRaises(ValueError):
                load_request_trace(_write(root, base))

    def test_configuration_requires_trace_for_request_trace(self):
        with self.assertRaises(ValueError):
            validate_config({"workload": "request_trace"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.json"
            path.write_text("{}", encoding="utf-8")
            config = validate_config({"workload": "request_trace", "trace_path": str(path)})
            self.assertEqual(config["trace_path"], str(path))
            with self.assertRaises(ValueError):
                validate_config(config, require_paths=True)


if __name__ == "__main__":
    unittest.main()
