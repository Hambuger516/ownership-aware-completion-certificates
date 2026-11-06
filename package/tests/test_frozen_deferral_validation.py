"""Replay unmodified real receipt excerpts inside bounded summary fixtures.

Fixture surrounding run records are synthetic and are not GPU measurements.
The two real excerpts retain source path/hash/index metadata in the JSON asset.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from prc_nc.analysis import summarize
import test_analysis_runner as old_fixtures


REAL=json.loads((Path(__file__).parent/"fixtures/real_frozen_deferral_receipts.json").read_text())


def fixture(folder, case=0):
    helper=old_fixtures.ReplayTests()
    events=[e for e in helper.fixture(folder) if e["event"]!="snapshot"]
    raw=copy.deepcopy(REAL[case])
    manifest=json.loads((folder/"manifest.json").read_text())
    manifest["configuration"]=raw["configuration"]|dict(count=1,waves=1)
    manifest["physical_capacity_pages"]=raw["physical_capacity_pages"]
    (folder/"manifest.json").write_text(json.dumps(manifest))
    return events+raw["events"]


def write(folder, events):
    (folder/"allocator_events.jsonl").write_text("\n".join(map(json.dumps,events)))


def event(events, kind):
    return next(e for e in events if e["event"]==kind)


class FrozenDeferralTests(unittest.TestCase):
    def test_both_real_audit_receipt_excerpts_pass_full_summary(self):
        for case in range(len(REAL)):
            with self.subTest(run=REAL[case]["source_run"]), tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);events=fixture(folder,case);write(folder,events)
                result,_,_=summarize(folder)
                self.assertEqual(result["status"],"PASS")
                self.assertEqual(result["verified_frozen_witness_deferrals"],1)

    def test_missing_duplicate_or_unbound_receipt_is_rejected(self):
        for defect in ("missing","duplicate","version","missing_version","action","view","source_order","flag_only"):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);events=fixture(folder)
                receipt=event(events,"drain_frozen_witness_check")
                if defect=="missing":events.remove(receipt)
                elif defect=="duplicate":events.insert(events.index(receipt)+1,copy.deepcopy(receipt))
                elif defect=="version":receipt["proposal_version"]+=1
                elif defect=="missing_version":receipt["proposal_version"]=None
                elif defect=="action":receipt["action_id"]+="-forged"
                elif defect=="view":receipt["active_view"]="C0_pinned_shared"
                elif defect=="source_order":receipt["source_event_index"]=0
                elif defect=="flag_only":
                    events=[e for e in events if e["event"] not in ("drain_frozen_witness_check","study_candidate_counterfactual")]
                write(folder,events)
                with self.assertRaises(AssertionError):summarize(folder)

    def test_forged_full_margins_or_projected_state_is_rejected(self):
        for defect in ("full_margin","min_margin","capacity","missing_post","cf_post","cf_hash","missing_cf"):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);events=fixture(folder)
                receipt=event(events,"drain_frozen_witness_check");cf=event(events,"study_candidate_counterfactual")
                if defect=="full_margin":receipt["c1_margins"][0][0]-=1
                elif defect=="min_margin":receipt["c1_min_margin"][0]-=1
                elif defect=="capacity":receipt["post"]["owner_classes"][0]["pages"]+=1
                elif defect=="missing_post":del receipt["post"]
                elif defect=="cf_post":cf["projected_post"]["free"][0]+=1
                elif defect=="cf_hash":cf["projected_post_sha256"]="0"*64
                elif defect=="missing_cf":events.remove(cf)
                write(folder,events)
                with self.assertRaises(AssertionError):summarize(folder)

    def test_refused_action_cannot_have_grant_or_allocation(self):
        for defect in ("receipt_grant","causal_grant","allocation"):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);events=fixture(folder)
                receipt=event(events,"drain_frozen_witness_check")
                if defect=="receipt_grant":receipt["grant_id"]=receipt["action_id"]+":grant"
                elif defect=="causal_grant":events.append(dict(event="causal_authorize",step=receipt["step"],
                    request_id=receipt["request_id"],action_id=receipt["action_id"],grant_id=receipt["action_id"]+":grant"))
                else:events.append(dict(event="allocation",step=receipt["step"],request_id=receipt["request_id"],
                    certificate=True,projection_verified=True))
                write(folder,events)
                with self.assertRaises(AssertionError):summarize(folder)

    def test_positive_head_and_real_result_binding_are_required(self):
        for defect in ("head_zero","follower_scheduled","missing_result","result_omits_head","result_order","missing_frame"):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);events=fixture(folder)
                receipt=event(events,"drain_frozen_witness_check")
                batch=event(events,"drain_batch");result=event(events,"drain_batch_confirmed")
                if defect=="head_zero":batch["scheduled_tokens"][receipt["head"]]=0
                elif defect=="follower_scheduled":batch["scheduled_tokens"][receipt["request_id"]]=1
                elif defect=="missing_result":events.remove(result)
                elif defect=="result_omits_head":result["returned_request_ids"].remove(receipt["head"])
                elif defect=="result_order":result["frozen_order"]=list(reversed(result["frozen_order"]))
                elif defect=="missing_frame":events.remove(event(events,"drain_head_first"))
                write(folder,events)
                with self.assertRaises(AssertionError):summarize(folder)

    def test_timing_receipt_scope_is_lifecycle_not_full_state_replay(self):
        for forged_summary in (False,True):
            with self.subTest(forged_summary=forged_summary),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);events=fixture(folder)
                manifest=json.loads((folder/"manifest.json").read_text())
                manifest["configuration"]["audit"]=False
                (folder/"manifest.json").write_text(json.dumps(manifest))
                receipt=event(events,"drain_frozen_witness_check")
                for key in ("post","c0_margins","c1_margins"):del receipt[key]
                events.remove(event(events,"study_candidate_counterfactual"))
                if forged_summary:receipt["c1_min_margin"]=[0]
                write(folder,events)
                if forged_summary:
                    with self.assertRaises(AssertionError):summarize(folder)
                else:self.assertEqual(summarize(folder)[0]["verified_frozen_witness_deferrals"],1)

    def test_existing_execution_check_still_runs_after_receipt_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);events=fixture(folder);write(folder,events)
            manifest=json.loads((folder/"manifest.json").read_text());manifest["execution_measurement"]=True
            (folder/"manifest.json").write_text(json.dumps(manifest))
            (folder/"execution_tokens.json").write_text(json.dumps(dict(pending_batches=1,executed_tokens=1,
                reexecuted_tokens=0,requests={"w0-r0":dict(executed_tokens=1,reexecuted_tokens=0)})))
            with self.assertRaisesRegex(AssertionError,"Unconfirmed execution batches"):
                summarize(folder)


if __name__=="__main__":unittest.main(verbosity=2)
