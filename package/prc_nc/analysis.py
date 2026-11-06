"""Independent replay of real page snapshots and contract-aware run summaries."""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
from pathlib import Path
from .algorithms import State, Group, indexed_closure
from .diagnostics import boundary_report
from .model import margins as reference_margins, closure as reference_closure
from .admission import AdmissionContext, decide



_NO_FILTER=object()


class _EventIndex:
    """Ordered multimaps of raw file positions; duplicates are never overwritten.

    Range lookups are exclusive at both endpoints, matching events[start+1:stop].
    Identity lookup retains the first position, matching the original `next(is)`.
    Only post-validation changes; no runtime certificate/model checks are skipped.
    """
    def __init__(self,events):
        self.events=events
        self.by_kind=defaultdict(list);self.by_action=defaultdict(list)
        self.by_grant=defaultdict(list);self.by_step=defaultdict(list)
        self.by_request_step=defaultdict(list);self.bad_request_step=defaultdict(list)
        self.missing_fields={};self.first_identity={};self.barriers=[]
        for i,e in enumerate(events):
            kind=e["event"];self.by_kind[kind].append(i)
            self.first_identity.setdefault(id(e),i)
            # These maps intentionally contain ordered LISTS, including repeats.
            if kind in ("causal_authorize","study_candidate_counterfactual"):
                self.by_action[kind,e.get("action_id")].append(i)
            if kind in ("causal_authorize","causal_execute","causal_acknowledge"):
                self.by_grant[kind,e.get("grant_id")].append(i)
            if kind in ("causal_authorize","causal_execute","causal_acknowledge","proposal"):
                self.by_step[kind,e.get("step")].append(i)
            if kind in ("incremental_proposal","allocation","causal_authorize","causal_execute","causal_acknowledge"):
                if "step" in e and "request_id" in e:
                    self.by_request_step[kind,(e["step"],e["request_id"])].append(i)
                else:
                    self.bad_request_step[kind].append(i)
                    self.missing_fields[i]="step" if "step" not in e else "request_id"
            if kind in ("proposal","incremental_proposal","allocation"):self.barriers.append(i)

    def positions(self,kind,*,action=_NO_FILTER,grant=_NO_FILTER,request_step=_NO_FILTER,step=_NO_FILTER,start=-1,stop=None):
        if request_step is not _NO_FILTER:
            bad=self.bad_request_step.get(kind,())
            a=bisect_right(bad,start);b=len(bad) if stop is None else bisect_left(bad,stop)
            if a<b:raise KeyError(self.missing_fields[bad[a]])
            rows=self.by_request_step.get((kind,request_step),())
        elif action is not _NO_FILTER:rows=self.by_action.get((kind,action),())
        elif grant is not _NO_FILTER:rows=self.by_grant.get((kind,grant),())
        elif step is not _NO_FILTER:rows=self.by_step.get((kind,step),())
        else:rows=self.by_kind.get(kind,())
        a=bisect_right(rows,start);b=len(rows) if stop is None else bisect_left(rows,stop)
        return rows[a:b]

    def matches(self,kind,**filters):return [(i,self.events[i]) for i in self.positions(kind,**filters)]
    def values(self,kind,**filters):return [self.events[i] for i in self.positions(kind,**filters)]
    def first_after(self,kind,start):
        rows=self.by_kind.get(kind,());at=bisect_right(rows,start)
        return rows[at] if at<len(rows) else None
    def next_barrier(self,start):
        at=bisect_right(self.barriers,start)
        return self.barriers[at] if at<len(self.barriers) else len(self.events)


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    return values[round((len(values) - 1) * q)] if values else None


def _verified_frozen_deferrals(events, config, capacity, *, _event_index=None):
    """Verify the new refusal receipt, without treating a flag as evidence.

    Audit logs permit independent reference-margin replay. Timing logs contain
    summaries only: their checks are receipt/lifecycle validation, not full
    independent state replay. Original validation below remains in force.
    """
    event_index=_event_index or _EventIndex(events)
    candidates = [i for i,e in event_index.matches("incremental_proposal")
                  if e.get("frozen_witness_defer") is True]
    if not candidates:
        return set()
    assert (config.get("drain_batching") == "witness_preserving"
            and config.get("drain_scope") == "cohort" and config.get("drain_policy") == "timer"
            and config["method"] == "shared_incremental"), "Frozen deferral outside configured policy"
    frames, batches, confirmed = {}, {}, {}
    actions, grants = set(), set()
    for e in events:
        kind = e["event"]
        if kind in ("drain_head_first", "drain_batch", "drain_batch_confirmed"):
            target = frames if kind == "drain_head_first" else batches if kind == "drain_batch" else confirmed
            key = (e["episode"], e["batch_index"])
            assert key not in target, "Duplicate frozen batch receipt"
            target[key] = e
        if kind.startswith("causal_"):
            if e.get("action_id"):
                actions.add(e["action_id"])
            if e.get("grant_id"):
                grants.add(e["grant_id"])
    verified, used_actions = set(), set()
    allocated = {(e["step"], e["request_id"]) for e in events if e["event"] == "allocation"}
    for i in candidates:
        inc = events[i]
        assert (inc.get("oracle_checked") is False and inc.get("fixed_accepted") is False
                and inc.get("full_accepted") is False and inc.get("equivalent") is None
                and inc.get("statistical_defer") is False), "Inconsistent frozen deferral flags"
        j = event_index.next_barrier(i)
        assert j < len(events) and events[j]["event"] == "proposal", "Frozen deferral missing final proposal"
        proposal = events[j]
        key = (inc["step"], inc["request_id"])
        assert (proposal["step"], proposal["request_id"]) == key, "Frozen deferral proposal identity mismatch"
        assert (proposal.get("allow") is False and proposal.get("certificate") is False
                and proposal.get("physical_fit") is True and key not in allocated), "Frozen refusal was allocated"
        receipts = event_index.values("drain_frozen_witness_check",start=i,stop=j)
        assert len(receipts) == 1, "Frozen deferral needs one unique fixed-witness receipt"
        receipt = receipts[0]
        assert (receipt["step"], receipt["request_id"]) == key, "Frozen receipt identity mismatch"
        assert (type(receipt.get("proposal_version")) is int and receipt["proposal_version"] >= 0
                and receipt["proposal_version"] == inc.get("authority_version")), "Frozen receipt version mismatch"
        action = receipt.get("action_id")
        assert isinstance(action, str) and action and action not in used_actions, "Frozen receipt action missing/duplicated"
        used_actions.add(action)
        assert (receipt.get("grant_id") is None and action not in actions
                and action + ":grant" not in grants), "Refused frozen action has a causal grant/dispatch"
        assert (receipt.get("fixed_accepted") is False and receipt.get("actual_allow") is False
                and receipt.get("reason") == "frozen_witness_negative_margin"), "Invalid frozen refusal receipt"
        frame_key = (receipt["episode"], receipt["batch_index"])
        frame, batch, result = frames.get(frame_key), batches.get(frame_key), confirmed.get(frame_key)
        assert frame is not None and batch is not None and result is not None, "Frozen refusal lacks batch/result binding"
        frozen, head = receipt["frozen_order"], receipt["head"]
        assert (frozen and len(set(frozen)) == len(frozen) and head == frozen[0]
                and inc["request_id"] in frozen and inc["request_id"] != head), "Frozen refusal is not a protected follower"
        assert (frame["step"] == batch["step"] == inc["step"] and frame["head"] == batch["head"] == head
                and frame["frozen_order"] == batch["frozen_order"] == frozen), "Frozen batch/order mismatch"
        assert frame["source_event_index"] < inc["source_event_index"] < receipt["source_event_index"] < proposal["source_event_index"] < batch["source_event_index"], "Frozen receipt event order mismatch"
        tokens = batch["scheduled_tokens"]
        assert (batch.get("batching") == "witness_preserving" and batch.get("empty") is False
                and type(receipt.get("head_tokens_scheduled")) is int
                and receipt["head_tokens_scheduled"] > 0
                and tokens.get(head) == receipt["head_tokens_scheduled"]
                and inc["request_id"] not in tokens), "Frozen refusal lacks actual positive-head batch budget"
        assert (result.get("physical_execution_observed") is True and result["head"] == head
                and result.get("batching") == "witness_preserving" and result["frozen_order"] == frozen
                and result["scheduled_tokens"] == tokens and set(tokens) <= set(result["returned_request_ids"])), "Frozen refusal lacks confirmed model-result receipt"
        c0_active = config.get("certificate_variant") in ("common_full_c0", "exclusive_only")
        assert receipt["active_view"] == ("C0_pinned_shared" if c0_active else "C1_physical"), "Frozen receipt active-view mismatch"
        for label in ("c0", "c1"):
            minimum = receipt[label + "_min_margin"]
            assert (len(minimum) == 1 and all(type(v) is int for v in minimum)
                    and receipt[label + "_fixed_accepted"] is all(v >= 0 for v in minimum)), "Frozen margin summary mismatch"
        active = "c0" if c0_active else "c1"
        assert not (receipt["c0_fixed_accepted"] and not receipt["c1_fixed_accepted"]), "Conservative frozen view exceeds physical view"
        assert receipt[active + "_fixed_accepted"] is False, "Frozen refusal has no negative active margin"
        post = receipt.get("post")
        if config.get("audit"):
            assert post is not None, "Audit frozen deferral missing projected physical state"
        if post is not None:
            ids = post["cohort"]
            assert len(set(ids)) == len(ids) and set(ids) == set(frozen), "Frozen receipt cohort mismatch"
            state = State(tuple(post["free"]), tuple(map(tuple, post["need"])), tuple(
                Group(str(k), g["pool"], g["pages"], frozenset(g["owner_ranks"]), g["pinned"])
                for k, g in enumerate(post["owner_classes"])))
            assert state.n == len(ids) and state.capacity == (capacity,), "Frozen projected capacity mismatch"
            assert proposal["free_after"] == state.free[0], "Frozen proposal/post free mismatch"
            rank = {rid:k for k, rid in enumerate(ids)}
            order = tuple(rank[rid] for rid in frozen)
            for label, conservative in (("c0", True), ("c1", False)):
                view = State(state.free, state.need, tuple(Group(g.key,g.pool,g.count,g.owners,
                    g.pinned or (conservative and len(g.owners)>1)) for g in state.groups))
                margins = reference_margins(view, order)
                minimum = tuple(min(row[p] for row in margins) for p in range(state.p))
                assert tuple(map(tuple, receipt[label + "_margins"])) == margins, "Frozen full margins differ from reference replay"
                assert tuple(receipt[label + "_min_margin"]) == minimum, "Frozen minimum differs from reference replay"
        cf = event_index.values("study_candidate_counterfactual",start=i,stop=j)
        if config.get("audit"):
            assert len(cf) == 1, "Audit frozen refusal missing unique counterfactual binding"
        if cf:
            assert len(cf) == 1 and cf[0].get("action_id") == action and cf[0].get("proposal_version") == receipt["proposal_version"], "Frozen counterfactual action/version mismatch"
            assert (cf[0]["step"], cf[0]["request_id"]) == key, "Frozen counterfactual identity mismatch"
            assert (cf[0].get("actual_allow") is False and cf[0].get("actual_grant_id") is None
                    and cf[0].get("projected_post") == post), "Frozen counterfactual decision/post mismatch"
            digest = hashlib.sha256(json.dumps(post, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert cf[0].get("projected_post_sha256") == digest, "Frozen counterfactual post hash mismatch"
        verified.add(id(inc))
    return verified



def _verify_ordinary_priority(events, config, capacity):
    """Receipt-only ordinary rank reuse; zero head work is a valid observation."""
    counts=dict(observed_ordinary_steps=0,relative_order_changes=0,confirmed_batches=0,
                activated_workload_reorders=0,positive_head_work_batches=0)
    if not config.get("ordinary_witness_priority",False):return counts
    assert config.get("ordinary_witness_priority") is True and config.get("rank_order") is False
    assert (config.get("drain_scope"),config.get("drain_batching"),config.get("drain_policy")) == ("cohort","cohort_certified","timer")
    frames,budgets,confirmed,snapshots={},{},{},{}
    active=False;pending=None;previous=None
    for ordinal,e in enumerate(events):
        assert e.get("source_event_index")==ordinal, "Ordinary observation file ordinal changed"
        mono=e.get("monotonic_ns")
        assert type(mono)is int and (previous is None or previous<=mono), "Ordinary observation clock invalid"
        previous=mono;kind=e["event"]
        if kind=="drain_start":
            assert pending is None, "Drain started before ordinary model confirmation"
            active=True
        elif kind=="drain_segment_end":active=False
        elif kind=="snapshot":snapshots[e["step"]]=e
        elif kind=="ordinary_witness_priority":
            assert not active and pending is None, "Ordinary sort overlaps drain or unconfirmed model"
            step=e["step"];assert step not in frames, "Duplicate ordinary cut"
            frames[step]=(ordinal,e)
            ids=e["resident_ids"];witness=e["witness"];before=e["prior_running"]
            assert len(set(ids))==len(ids) and len(witness)==len(ids) and set(witness)==set(ids)
            rank={rid:i for i,rid in enumerate(witness)}
            expected=sorted(before,key=lambda rid:rank.get(rid,len(rank)))
            assert e["proposed_running"]==e["reordered_running"]==expected, "Ordinary full stable rank sort differs"
            assert e["head"]==(witness[0] if witness else None)
            assert e["head_running_count"]==(before.count(e["head"]) if e["head"] is not None else 0)
            assert e["relative_order_changed"] is (before!=expected), "Ordinary reorder activation was invented"
            assert e["drain_active_at_cut"] is False and e["pending_grant"] is False and e["resource_states_equal"] is True
            assert e["pending_model_batches"]==e["pending_meter_batches"]==0
            assert type(e["authority_version_at_cut"])is int and e["authority_version_at_cut"]>=0
            counts["observed_ordinary_steps"]+=1;counts["relative_order_changes"]+=int(before!=expected)
        elif kind in ("ordinary_witness_budget","ordinary_witness_confirmed"):
            step=e["step"];target=budgets if kind=="ordinary_witness_budget" else confirmed
            assert step in frames and step not in target, "Unknown/duplicate ordinary budget/model receipt"
            target[step]=(ordinal,e);frame=frames[step][1]
            for key in set(frame)-{"event","source_event_index","monotonic_ns"}:assert e[key]==frame[key], "Ordinary model/cut identity differs"
            tokens=e["scheduled_tokens"]
            assert isinstance(tokens,dict) and all(type(v)is int and v>0 for v in tokens.values())
            assert e["empty"] is (not bool(tokens)) and e["head_tokens"]==tokens.get(frame["head"],0)
            assert e["scheduled_outside_initial_residents"]==sorted(set(tokens)-set(frame["resident_ids"]))
            # Intentionally no closed-cohort subset check: ordinary admission is open.
            if kind=="ordinary_witness_budget":
                if tokens:pending=step
            else:
                assert pending==step and step in budgets and tokens==budgets[step][1]["scheduled_tokens"]
                assert e["physical_execution_observed"] is True and set(tokens)<=set(e["returned_request_ids"])
                observed=e["confirmed_executed_positions"];repeated=e["confirmed_reexecuted_positions"];unique=e["confirmed_new_unique_positions"]
                assert observed==tokens and set(repeated)==set(unique)==set(tokens)
                assert all(type(repeated[r])is int and 0<=repeated[r]<=observed[r] and unique[r]==observed[r]-repeated[r] for r in tokens)
                head=frame["head"];head_unique=unique.get(head,0)
                assert e["head_new_unique_positions"]==head_unique
                assert e["head_positive_unique_execution"] is (frame["head_running_count"]==1 and head_unique>0), "Zero/ambiguous head work credited"
                workload_unique=sum(v for r,v in unique.items() if r.startswith("w") and not r.startswith("warmup"))
                counts["activated_workload_reorders"]+=int(frame["relative_order_changed"] and workload_unique>0)
                counts["positive_head_work_batches"]+=int(e["head_positive_unique_execution"])
                counts["confirmed_batches"]+=1;pending=None
    assert pending is None and frames.keys()==budgets.keys(), "Missing ordinary budget/model settlement"
    for step,(fi,frame) in frames.items():
        bi,budget=budgets[step];assert fi<bi
        if budget["scheduled_tokens"]:assert step in confirmed and bi<confirmed[step][0]
        else:assert step not in confirmed, "Empty ordinary step fabricated model work"
        if config.get("audit"):
            physical=snapshots.get(step);assert physical is not None and set(physical["cohort"])==set(frame["resident_ids"])
            ids=physical["cohort"];rank={rid:i for i,rid in enumerate(ids)}
            conservative=config.get("certificate_variant")=="common_full_c0"
            groups=tuple(Group(str(g["block"]),0,1,frozenset(rank[r] for r in g["owners"]),conservative and len(g["owners"])>1) for g in physical["groups"])
            state=State((physical["free"],),tuple((n,) for n in physical["hard_need"]),groups)
            assert state.capacity==(capacity,)
            assert all(v>=0 for row in reference_margins(state,tuple(rank[r] for r in frame["witness"])) for v in row)
    return counts


def _verify_cohort_certified_batches(events, config, capacity):
    """Replay closed-membership/epoch/model receipts for the new policy.

    Audit mode also reference-checks the actual epoch witness on the physical
    snapshot. Timing mode verifies receipts only, not an unavailable full-state
    replay. Real GPU execution is additionally counted by the execution meter.
    """
    if config.get("active_drain_admission",False):
        from .active_admission_analysis import verify_active_admission
        return verify_active_admission(events,config,capacity)
    if config.get("drain_batching") != "cohort_certified":
        return
    assert config.get("drain_scope") in ("cohort", "first_completion") and config.get("drain_policy") == "timer"
    first_scope = config.get("drain_scope") == "first_completion"
    if events:
        assert type(capacity) is int and capacity >= 0, "Cohort replay requires measured physical capacity"
    expected_semantics = "current_epoch_authority_witness; membership_frozen_until_confirmed_terminal"
    frames, heads, batches, results, snapshots = {}, {}, {}, {}, {}
    active, episode = set(), None
    previous_mono = None
    closed_open = False
    current_order, current_key = [], None
    first_success, last_terminal, resume_pending = None, None, None
    early_ends = []
    expected_head_cancel = None
    for ordinal, e in enumerate(events):
        assert e.get("source_event_index") == ordinal, "Cohort event ordinal differs from actual file order"
        mono = e.get("monotonic_ns")
        assert type(mono) is int and (previous_mono is None or mono >= previous_mono), "Cohort event clock/order invalid"
        previous_mono = mono
        kind = e["event"]
        if first_scope and expected_head_cancel is not None:
            assert kind not in ("causal_finish", "drain_segment_end", "drain_epoch_witness", "drain_batch_confirmed"), "Missing epoch-head cancellation receipt"
        if first_scope and resume_pending is not None:
            assert kind not in ("drain_start", "drain_epoch_witness", "proposal", "causal_authorize", "allocation"), "New admission/work preceded prior closing model confirmation"
        if kind == "drain_start":
            assert not active and (not first_scope or not closed_open), "Overlapping closed cohorts"
            ids = e["cohort"]
            assert ids and len(set(ids)) == len(ids), "Invalid initial closed cohort"
            active, episode = set(ids), None
            if first_scope:
                closed_open = True
                current_order, current_key = list(ids), None
                first_success, last_terminal = None, None
        elif kind == "causal_finish" and e.get("request_id") in active:
            assert e.get("outcome") in ("completed", "cancelled") and e.get("release_observed") is True
            active.remove(e["request_id"])
            if first_scope:
                last_terminal = (ordinal, e)
                current_order.remove(e["request_id"])
                if e["outcome"] == "cancelled" and current_key is not None and e["request_id"] == frames[current_key][1]["head"]:
                    expected_head_cancel = (ordinal, e)
                if e["outcome"] == "completed":
                    assert first_success is None, "First successful callback did not immediately end segment"
                    first_success = (ordinal, e, current_key)
        elif kind == "drain_head_cancelled" and first_scope:
            assert expected_head_cancel is not None, "Nonhead/unknown cancellation falsely marked epoch head"
            ti, terminal = expected_head_cancel
            assert ti < ordinal and e.get("request_id") == terminal["request_id"] and e["step"] == terminal["step"], "Epoch-head cancellation callback binding differs"
            assert e.get("successful_completion") is False and e.get("next_frozen_head") is None and e.get("replacement_pending") is False, "Cancelled epoch head was replaced before settled next epoch"
            expected_head_cancel = None
        elif kind == "drain_segment_end":
            if not first_scope:
                assert not active and e.get("remaining_frozen_requests") == [], "Closed cohort ended before confirmed terminals"
            else:
                assert closed_open and e.get("scope") == "first_completion" and e.get("batching") == "cohort_certified", "First-completion end outside actual segment"
                assert e.get("episode") == episode and e.get("remaining_frozen_requests") == current_order and set(current_order) == active, "First-completion survivors/episode differ from confirmed callbacks"
                if first_success is not None:
                    ti, terminal, key = first_success
                    assert key is not None and e.get("reason") == "first_member_completed", "First-completion end missing actual successful member"
                    assert e.get("completed_request_id") == terminal["request_id"] and terminal["step"] == e["step"], "First-completion winner/callback binding differs"
                    assert e.get("successful_member_completions") == 1, "Successful member count differs"
                    assert e.get("successful_head_completions") == int(terminal["request_id"] == frames[key][1]["head"]), "Nonhead winner credited as head"
                    early_ends.append((ordinal, ti, terminal, key))
                else:
                    assert not active and last_terminal is not None and last_terminal[1]["outcome"] == "cancelled", "No successful terminal or complete cancellation set"
                    assert e.get("reason") == "cohort_empty_without_member_completion" and e.get("completed_request_id") == last_terminal[1]["request_id"], "Cancellation falsely ended successfully"
                    assert e.get("successful_member_completions") == e.get("successful_head_completions") == 0, "Cancellation credited as success"
                resume_pending = current_key if current_key not in results else None
                active, closed_open = set(), False
        elif kind == "snapshot":
            assert e["step"] not in snapshots, "Duplicate epoch physical snapshot"
            snapshots[e["step"]] = e
        elif kind in ("drain_epoch_witness", "drain_head_first", "drain_batch", "drain_batch_confirmed"):
            key = (e["episode"], e["batch_index"])
            target = {"drain_epoch_witness":frames, "drain_head_first":heads,
                      "drain_batch":batches, "drain_batch_confirmed":results}[kind]
            assert key not in target, "Duplicate cohort epoch/model receipt"
            target[key] = (ordinal, e)
            assert e.get("order_semantics") == expected_semantics, "Ambiguous mutable witness semantics"
            if kind == "drain_epoch_witness":
                assert set(frames) - {key} == set(results), "Epoch refresh preceded prior model confirmation"
                assert active and (episode is None or episode == e["episode"]), "Epoch outside closed cohort"
                episode = e["episode"]
                order = e["epoch_order"]
                if first_scope:
                    assert closed_open and first_success is None, "Epoch follows an unclosed successful callback"
                    current_order, current_key = list(order), key
                assert order and len(set(order)) == len(order) and set(order) == active, "Epoch changed frozen membership"
                assert e["cohort_members"] == sorted(active) and e["head"] == order[0]
                assert e.get("pending_grant") is False and e.get("prior_model_batch_pending") is False
                assert type(e.get("authority_version")) is int and e["authority_version"] >= 0
            if kind == "drain_batch_confirmed":
                assert key in frames and key in batches, "Model confirmation precedes actual epoch/batch"
                if first_scope and key == resume_pending:
                    resume_pending = None
    assert not active and (not first_scope or not closed_open), "Closed cohort has unconfirmed terminal members at end of run"
    if first_scope:
        assert resume_pending is None, "Closing model batch never confirmed"
        assert expected_head_cancel is None, "Missing final epoch-head cancellation receipt"
        for ei, ti, terminal, key in early_ends:
            assert key in batches and key in results, "Winner has no actual batch/model confirmation"
            bi, batch = batches[key]; ri, result = results[key]
            assert bi < ti < ei < ri, "Winner callback/end must precede closing model-result confirmation"
            assert terminal["step"] == batch["step"] and terminal["request_id"] in batch["scheduled_tokens"], "Winner was not executed by the closing model batch"
            assert result["confirmed_real_positions"].get(terminal["request_id"], 0) > 0, "Winner lacks real unique model work"
    assert frames.keys() == heads.keys() == batches.keys() == results.keys(), "Missing cohort epoch/head/batch/model receipt"
    for key, (fi, frame) in frames.items():
        hi, head = heads[key]; bi, batch = batches[key]; ri, result = results[key]
        assert fi < hi < bi < ri, "Cohort epoch/model receipt chronology invalid"
        order = frame["epoch_order"]
        for e in (head, batch, result):
            assert e["head"] == frame["head"] and e["frozen_order"] == order, "Epoch order/head binding mismatch"
            assert e["step"] == frame["step"], "Epoch/result step differs"
        assert (head["after"] == [frame["head"]] + [rid for rid in head["before"] if rid != frame["head"]]
                and head["before"].count(frame["head"]) == 1), "Head/follower priority was not stable"
        tokens = batch["scheduled_tokens"]
        assert batch.get("batching") == result.get("batching") == "cohort_certified"
        assert batch.get("empty") is False and tokens and set(tokens) <= set(order)
        assert all(type(v) is int and v > 0 for v in tokens.values()), "Nonpositive actual cohort token budget"
        assert batch["head_tokens"] == tokens.get(frame["head"], 0) > 0
        assert result.get("physical_execution_observed") is True and result["scheduled_tokens"] == tokens
        assert set(tokens) <= set(result["returned_request_ids"]), "Model result omitted scheduled member"
        assert result.get("confirmed_real_positions") == tokens and result.get("confirmed_new_unique_positions") == sum(tokens.values()) > 0, "No unique confirmed real execution"
        if config.get("audit"):
            physical = snapshots.get(frame["step"])
            assert physical is not None and set(physical["cohort"]) == set(order), "Missing epoch physical witness state"
            ids = physical["cohort"]; rank = {rid:i for i,rid in enumerate(ids)}
            conservative = config.get("certificate_variant") in ("common_full_c0", "exclusive_only")
            groups = tuple(Group(str(g["block"]), 0, 1, frozenset(rank[rid] for rid in g["owners"]),
                                 conservative and len(g["owners"]) > 1) for g in physical["groups"])
            state = State((physical["free"],), tuple((n,) for n in physical["hard_need"]), groups)
            assert state.capacity == (capacity,), "Epoch physical capacity mismatch"
            margins = reference_margins(state, tuple(rank[rid] for rid in order))
            assert all(v >= 0 for row in margins for v in row), "Current epoch Authority witness invalid"




def _verify_head_funded(events, config, capacity, *, _event_index=None):
    """Strict segment/cancellation/model/funding receipts; no global oracle fiction."""
    if config.get("drain_batching")!="head_funded":return set()
    assert config.get("drain_scope")=="head" and config.get("drain_policy")=="timer"
    if not events:return set()
    event_index=_event_index or _EventIndex(events)
    assert type(capacity) is int and capacity>=0
    semantics="persistent_selected_head; frozen_survivor_order_not_a_retained_witness"
    frames={};heads={};batches={};results={};snapshots={};funds=[]
    active=set();selected=None;cancel_pending=False;last_terminal=None;previous=None
    for i,e in enumerate(events):
        assert e.get("source_event_index")==i, "Head funding ordinal differs from file order"
        now=e.get("monotonic_ns");assert type(now) is int and (previous is None or previous<=now)
        previous=now;kind=e["event"]
        if kind=="drain_start":
            assert not active, "Overlapping head-funded segments"
            active=set(e["cohort"]);assert active and len(active)==len(e["cohort"])
            selected=None;cancel_pending=False;last_terminal=None
        elif kind=="causal_finish" and e.get("request_id") in active:
            assert e.get("release_observed") is True and e.get("outcome") in ("completed","cancelled")
            rid=e["request_id"];active.remove(rid)
            if rid==selected:
                last_terminal=(rid,e["outcome"])
                if e["outcome"]=="cancelled":selected=None;cancel_pending=True
        elif kind=="drain_segment_end":
            if e["reason"]=="selected_head_completed":
                assert last_terminal==(e["completed_request_id"],"completed") and selected==e["completed_request_id"]
                assert e["successful_head_completions"]==1 and set(e["remaining_frozen_requests"])==active
            else:
                assert e["reason"]=="cohort_empty_without_head_completion" and e["remaining_frozen_requests"]==[]
                assert not active and e["successful_head_completions"]==0, "Cancellation/empty segment credited completion"
            active=set();selected=None;cancel_pending=False
        elif kind=="snapshot":
            assert e["step"] not in snapshots;snapshots[e["step"]]=e
        elif kind in ("drain_head_funded_epoch","drain_head_first","drain_batch","drain_batch_confirmed"):
            key=(e["episode"],e["batch_index"])
            target={"drain_head_funded_epoch":frames,"drain_head_first":heads,"drain_batch":batches,"drain_batch_confirmed":results}[kind]
            assert key not in target;target[key]=(i,e)
            assert e.get("order_semantics")==semantics
            if kind=="drain_head_funded_epoch":
                assert set(frames)-{key}==set(results), "Head selection before prior model confirmation"
                assert active and set(e["frozen_survivors"])==active and len(e["frozen_survivors"])==len(active)
                assert e["head"]==e["selected_head"]==e["frozen_survivors"][0]
                witness=e["authority_witness"]
                assert len(witness)==len(active) and set(witness)==active
                if selected is None:
                    cause="selected_head_cancelled" if cancel_pending else "segment_start"
                    assert e["head_selection_cause"]==cause and e["head_replacement"] is cancel_pending
                    assert e["head"]==witness[0], "Replacement not chosen from current Authority witness"
                    selected=e["head"];cancel_pending=False
                else:
                    assert e["head"]==selected and e["head_selection_cause"]=="retained" and e["head_replacement"] is False, "Surviving selected head replaced"
                assert e["pending_grant"] is False and e["prior_model_batch_pending"] is False
                assert type(e["authority_version"]) is int and e["authority_version"]>=0
                assert tuple(e["margin"])==tuple(f-d for f,d in zip(e["post_free"],e["head_need"]))
                assert len(e["margin"])==len(e["post_free"])==len(e["head_need"]) and all(v>=0 for v in e["margin"])
            elif kind=="drain_batch_confirmed":assert key in frames and key in batches
        elif kind=="drain_head_funding_check":funds.append((i,e))
    assert not active, "Unresolved head-funded segment at end of run"
    assert frames.keys()==heads.keys()==batches.keys()==results.keys(), "Missing funded epoch/batch/model receipt"
    for key,(fi,frame) in frames.items():
        hi,head=heads[key];bi,batch=batches[key];ri,result=results[key]
        assert fi<hi<bi<ri
        order=frame["frozen_survivors"]
        for e in (head,batch,result):
            assert e["step"]==frame["step"] and e["head"]==frame["head"] and e["frozen_order"]==order
        assert head["before"].count(frame["head"])==1
        assert head["after"]==[frame["head"]]+[rid for rid in head["before"] if rid!=frame["head"]]
        tokens=batch["scheduled_tokens"]
        assert batch["batching"]==result["batching"]=="head_funded" and batch["empty"] is False
        assert tokens and set(tokens)<=set(order) and all(type(v) is int and v>0 for v in tokens.values())
        assert tokens.get(frame["head"],0)==batch["head_tokens"]==result["head_tokens"]>0
        assert result["physical_execution_observed"] is True and result["scheduled_tokens"]==tokens
        assert set(tokens)<=set(result["returned_request_ids"])
        assert result["confirmed_real_positions"]==tokens and result["confirmed_new_unique_positions"]==sum(tokens.values())
        if config.get("audit"):
            snapshot=snapshots.get(frame["step"]);assert snapshot is not None
            ids=snapshot["cohort"];rank={rid:i for i,rid in enumerate(ids)}
            assert set(ids)==set(order)
            conservative=config.get("certificate_variant")=="common_full_c0"
            state=State((snapshot["free"],),tuple((n,) for n in snapshot["hard_need"]),tuple(
                Group(str(g["block"]),0,1,frozenset(rank[rid] for rid in g["owners"]),conservative and len(g["owners"])>1)
                for g in snapshot["groups"]))
            assert state.capacity==(capacity,)
            assert all(v>=0 for row in reference_margins(state,tuple(rank[rid] for rid in frame["authority_witness"])) for v in row)
            h=rank[frame["head"]]
            assert tuple(frame["post_free"])==state.free and tuple(frame["head_need"])==state.need[h]
    verified=set();seen_actions=set();fund_keys=set();defer_keys=set();settled_by_frame={}
    for i,e in funds:
        key=(e["episode"],e["batch_index"]);assert key in frames
        fi,frame=frames[key];hi,_=heads[key];bi,batch=batches[key];ri,result=results[key]
        assert fi<hi<i<bi<ri and e["step"]==frame["step"]
        h=frame["head"];assert e["head"]==e["selected_head"]==h and e["frozen_survivors"]==frame["frozen_survivors"]
        assert len(set(e["post_cohort"]))==len(e["post_cohort"]) and set(e["post_cohort"])==set(frame["frozen_survivors"])
        free,need,margin=e["post_free"],e["head_need"],e["margin"]
        assert free and len(free)==len(need)==len(margin)
        assert all(type(v) is int for row in (free,need,margin) for v in row)
        assert tuple(margin)==tuple(f-d for f,d in zip(free,need))
        funded=all(v>=0 for v in margin);assert e["accepted"] is funded and e["authority_checked"] is funded
        assert type(e["proposal_version"]) is int and e["proposal_version"]>=0
        action=e["action_id"];assert isinstance(action,str) and action and action not in seen_actions;seen_actions.add(action)
        k=(e["step"],e["request_id"]);assert k not in fund_keys;fund_keys.add(k)
        inc=event_index.values("incremental_proposal",request_step=k,stop=i)
        assert len(inc)==1;inc=inc[0];assert inc["authority_version"]==e["proposal_version"]
        j=event_index.first_after("proposal",i)
        if j is None:raise StopIteration
        proposal=events[j];assert (proposal["step"],proposal["request_id"])==k
        assert proposal["physical_fit"] is True and proposal["free_after"]==free[0]
        assert proposal["allow"] is e["actual_allow"] and proposal["certificate"] is e["active_certificate"]
        assert inc.get("head_funding_defer") is (not funded) and proposal.get("head_funding_defer") is (not funded)
        grants=event_index.values("causal_authorize",action=action)
        if e["request_id"]!=h:
            assert e["head_tokens_scheduled"]==batch["scheduled_tokens"].get(h,0)>0
            assert h in settled_by_frame.get(key,{}), "Follower lacks a previously settled head grant"
            assert settled_by_frame[key][h]["ack_index"] < i, "Follower funding check preceded selected-head ACK"
        if not funded:
            assert e["request_id"]!=h and e["head_tokens_scheduled"]==batch["scheduled_tokens"].get(h,0)>0
            assert not grants and e["grant_id"] is None and e["actual_allow"] is False and e["active_certificate"] is False
            assert not any(event_index.positions(kind,grant=action+":grant")
                or event_index.positions(kind,request_step=k)
                for kind in ("causal_authorize","causal_execute","causal_acknowledge")), "Unfunded deferred action acquired/executed/acknowledged a grant"
            assert e["certificate_status"]=="not_evaluated_head_funding_defer"
            assert inc["oracle_checked"] is False and inc["equivalent"] is None and inc["fixed_accepted"] is False and inc["full_accepted"] is False
            assert inc.get("statistical_defer") is False and inc.get("frozen_witness_defer") is False
            assert e["request_id"] not in batch["scheduled_tokens"]
            verified.add(id(inc));defer_keys.add(k)
        elif e["actual_allow"]:
            assert e["active_certificate"] is True and e["certificate_status"]=="certified"
            assert len(grants)==1 and grants[0]["grant_id"]==e["grant_id"] and grants[0]["request_id"]==e["request_id"] and e["request_id"] in batch["scheduled_tokens"]
            grant=e["grant_id"]
            assert grant==action+":grant", "Head funding action/grant identity differs"
            authorize=event_index.matches("causal_authorize",grant=grant)
            execute=event_index.matches("causal_execute",grant=grant)
            acknowledge=event_index.matches("causal_acknowledge",grant=grant)
            allocation=event_index.matches("allocation",request_step=k)
            assert len(authorize)==len(execute)==len(acknowledge)==len(allocation)==1, "Allowed head-funding action lacks unique authorize/execute/ACK/projection chain"
            au_i,au=authorize[0];ex_i,ex=execute[0];ack_i,ack=acknowledge[0];al_i,al=allocation[0]
            inc_i=event_index.first_identity[id(inc)]
            assert hi < au_i < inc_i < i < j < ex_i < ack_i < al_i < bi, "Head-funding transaction chronology invalid"
            assert au["action_id"]==action and au["execution_confirmed"] is False
            for a,kind,version in ((au,"authorize",e["proposal_version"]+1),(ex,"execute",e["proposal_version"]+1),(ack,"acknowledge",e["proposal_version"]+2)):
                assert a["kind"]==kind and a["request_id"]==e["request_id"] and a["step"]==e["step"]
                assert type(a["state_version"]) is int and a["state_version"]==version, "Authority version differs in head-funding chain"
                assert a["worker_id"]==au["worker_id"] and a["pool_id"]==au["pool_id"]
            assert isinstance(au["worker_id"],str) and au["worker_id"] and isinstance(au["pool_id"],str) and au["pool_id"]
            for vector in (au["confirmed_free"],au["virtual_free"],ex["executed_pages"],ex["physical_free"],ack["acknowledged_pages"],ack["confirmed_free"],ack["virtual_free"]):
                assert len(vector)==len(free) and all(type(v) is int and v>=0 for v in vector)
            pages=tuple(au["authorized_pages"])
            assert len(pages)==len(free) and all(type(v) is int and v>=0 for v in pages)
            assert tuple(ex["executed_pages"])==tuple(ack["acknowledged_pages"])==pages
            assert tuple(au["virtual_free"])==tuple(ex["physical_free"])==tuple(ack["confirmed_free"])==tuple(ack["virtual_free"])==tuple(free), "Executed/ACK post-state free vector differs from certified projection"
            assert len(au["confirmed_free"])==len(free)
            assert tuple(before-after for before,after in zip(au["confirmed_free"],free))==pages
            assert au["confirmed_free"][0]==proposal["free_before"] and pages[0]==proposal["free_before"]-proposal["free_after"]
            assert al["projection_verified"] is True and al["certificate"] is True and al["free"]==free[0]
            if e["request_id"]!=h:
                assert settled_by_frame[key][h]["ack_index"] < au_i, "Follower was authorized before selected-head ACK"
            settled_by_frame.setdefault(key,{})[e["request_id"]]=dict(ack_index=ack_i,grant_id=grant)
        else:
            assert not grants and e["grant_id"] is None and e["certificate_status"]=="refused_active_view"
        cf=event_index.values("study_candidate_counterfactual",action=action,start=i,stop=j)
        if config.get("audit"):assert len(cf)==1 and e.get("projected_post") is not None
        if cf:
            assert len(cf)==1;cf=cf[0];post=e["projected_post"]
            assert cf["projected_post"]==post and cf["request_id"]==e["request_id"] and cf["proposal_version"]==e["proposal_version"]
            assert cf["actual_grant_id"]==e["grant_id"] and cf["actual_allow"] is e["actual_allow"]
            digest=hashlib.sha256(json.dumps(post,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
            assert e["projected_post_sha256"]==cf["projected_post_sha256"]==digest
            rank=post["cohort"].index(h)
            assert tuple(post["free"])==tuple(free) and tuple(post["need"][rank])==tuple(need) and post["cohort"]==e["post_cohort"]
            state=State(tuple(post["free"]),tuple(map(tuple,post["need"])),tuple(Group(str(k),g["pool"],g["pages"],frozenset(g["owner_ranks"]),g["pinned"]) for k,g in enumerate(post["owner_classes"])))
            assert state.capacity==(capacity,)
            c0=State(state.free,state.need,tuple(Group(g.key,g.pool,g.count,g.owners,g.pinned or len(g.owners)>1) for g in state.groups))
            assert cf["shared_certificate"] is (len(reference_closure(state))==state.n)
            assert cf["exclusive_certificate"] is (len(reference_closure(c0))==state.n)
            if funded:
                assert e["active_certificate"] is (cf["exclusive_certificate"] if config.get("certificate_variant")=="common_full_c0" else cf["shared_certificate"])
    for key,(_,batch) in batches.items():
        assert set(settled_by_frame.get(key,{}))==set(batch["scheduled_tokens"]), "Actual drain budget lacks a settled grant for every member"
        allowed_grants={v["grant_id"] for v in settled_by_frame.get(key,{}).values()}
        step=frames[key][1]["step"]
        for kind in ("causal_authorize","causal_execute","causal_acknowledge"):
            actual=event_index.values(kind,step=step)
            assert len(actual)==len(allowed_grants) and {a.get("grant_id") for a in actual}==allowed_grants, "Orphan/duplicate transaction receipt inside funded epoch"
    frame_steps={e["step"] for _,e in frames.values()}
    assert {(e["step"],e["request_id"]) for e in events if e["event"]=="proposal" and e["step"] in frame_steps and e.get("physical_fit") is True}==fund_keys, "Missing head-funding receipt"
    for kind in ("proposal","incremental_proposal"):
        assert {(e["step"],e["request_id"]) for e in events if e["event"]==kind and e.get("head_funding_defer") is True}==defer_keys
    return verified


def _verify_admission_policy(events, config, request_caps, *, _event_index=None):
    """Separate utility deferrals from infeasible states, including cap truth.

    The Authority attests immutable cap context; this replay compares it with
    the experiment's actual hard-cap manifest and checks C0/C1 independently
    on audit projected states. Timing receipts cannot replace full-state audit.
    """
    event_index=_event_index or _EventIndex(events)
    receipts=event_index.matches("admission_policy_decision")
    if not receipts:
        assert not any(e.get("admission_policy_defer") is True or e.get("admission_decision") is not None for e in events), "Unreceipted admission utility decision"
        return 0
    assert all(e.get("source_event_index")==i for i,e in enumerate(events)), "Admission replay event ordinal invalid"
    assert isinstance(request_caps,list) and request_caps, "Admission cap truth manifest missing"
    def hard_cap(rid):
        # vLLM0.29 InputProcessor.assign_request_id appends exactly8 chars
        # of utils.random_uuid() (lowercase hexadecimal); disabled randomizing
        # retains the exact external ID. No broad prefix matching is allowed.
        if re.fullmatch(r"warmup(?:-[0-9a-f]{8})?",rid):return 8
        match=re.match(r"^w[0-9]+-r([0-9]+)(?:-|$)",rid)
        assert match and int(match.group(1))<len(request_caps), "Admission cap identity absent from actual contract"
        return request_caps[int(match.group(1))]
    resident=set();action_ids=set();verified=0;deferred_keys=set()
    for i,event in enumerate(events):
        kind=event["event"]
        if kind=="causal_acknowledge":resident.add(event["request_id"])
        elif kind=="causal_finish":resident.discard(event["request_id"])
        if kind!="admission_policy_decision":continue
        raw=event["decision"];ctx_raw=raw["context"]
        context=AdmissionContext(ctx_raw["candidate_id"],ctx_raw["authority_version"],ctx_raw["candidate_cap"],
            tuple(tuple(pair) for pair in ctx_raw["incumbent_caps"]))
        assert context.candidate_id==event["request_id"] and context.authority_version==event["proposal_version"]
        assert context.candidate_id not in resident and {rid for rid,_ in context.incumbent_caps}==resident, "Admission incumbent set differs from confirmed physical lifecycle"
        assert context.candidate_cap==hard_cap(context.candidate_id)
        assert all(cap==hard_cap(rid) for rid,cap in context.incumbent_caps), "Admission cap attestation differs from actual hard caps"
        replay=decide(config.get("admission_policy","unrestricted"),context,raw["c0_complete"],raw["c1_complete"])
        assert json.loads(json.dumps(replay.summary()))==raw, "Admission utility predicate summary invalid"
        action=event["action_id"]
        assert isinstance(action,str) and action and action not in action_ids, "Admission action missing/duplicated"
        action_ids.add(action)
        active = raw["c0_complete"] if config.get("certificate_variant")=="common_full_c0" else raw["c1_complete"]
        assert event["active_certificate"] is active
        deferred=active and not raw["policy_allows"]
        assert event["policy_deferred"] is deferred and event["actual_allow"] is (active and raw["policy_allows"])
        before=event_index.values("incremental_proposal",request_step=(event["step"],event["request_id"]),stop=i)
        assert len(before)==1, "Admission decision lacks unique algorithm proposal"
        inc=before[0]
        assert inc["authority_version"]==context.authority_version and inc.get("admission_policy_defer") is deferred
        assert inc["full_accepted"] is active and inc["oracle_checked"] is True and inc.get("equivalent") is True
        assert inc.get("statistical_defer") is False and inc.get("frozen_witness_defer") is False
        proposal_index=event_index.first_after("proposal",i)
        following=[] if proposal_index is None else [events[proposal_index]]
        assert len(following)==1
        proposal=following[0]
        assert (proposal["step"],proposal["request_id"])==(event["step"],event["request_id"])
        assert proposal["certificate"] is active and proposal["allow"] is event["actual_allow"]
        assert proposal.get("admission_policy_defer") is deferred
        grants=event_index.values("causal_authorize",action=action)
        if event["actual_allow"]:
            assert len(grants)==1 and grants[0]["grant_id"]==event["grant_id"] and grants[0].get("admission_decision")==raw
        else:
            assert not grants and event["grant_id"] is None, "Deferred/infeasible candidate received a grant"
        cf=event_index.values("study_candidate_counterfactual",action=action,start=i,stop=proposal_index)
        if config.get("audit"):assert len(cf)==1, "Audit utility receipt lacks exact projected-state counterfactual"
        if cf:
            assert len(cf)==1
            c=cf[0];post=c["projected_post"];ids=post["cohort"]
            assert c["request_id"]==context.candidate_id and c["proposal_version"]==context.authority_version, "Counterfactual candidate/version differs from policy decision"
            assert c["actual_grant_id"]==event["grant_id"], "Counterfactual grant differs from actual policy grant"
            digest=hashlib.sha256(json.dumps(post,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
            assert c["projected_post_sha256"]==digest, "Counterfactual projected-state hash mismatch"
            assert set(ids)==resident|{context.candidate_id} and c["admission_decision"]==raw
            assert c.get("admission_policy_defer") is deferred and c["actual_allow"] is event["actual_allow"]
            state=State(tuple(post["free"]),tuple(map(tuple,post["need"])),tuple(
                Group(str(k),g["pool"],g["pages"],frozenset(g["owner_ranks"]),g["pinned"])
                for k,g in enumerate(post["owner_classes"])))
            c0=State(state.free,state.need,tuple(Group(g.key,g.pool,g.count,g.owners,g.pinned or len(g.owners)>1) for g in state.groups))
            assert (len(reference_closure(c0))==state.n) is raw["c0_complete"]
            assert (len(reference_closure(state))==state.n) is raw["c1_complete"]
            assert c["exclusive_certificate"] is raw["c0_complete"] and c["shared_certificate"] is raw["c1_complete"]
        if deferred:
            verified+=1;deferred_keys.add((event["step"],event["request_id"]))
    assert {e.get("action_id") for e in events if e["event"] in ("causal_authorize","study_candidate_counterfactual") and e.get("admission_decision") is not None} <= action_ids, "Missing admission policy receipt for an actual action"
    for kind in ("incremental_proposal","proposal"):
        assert {(e["step"],e["request_id"]) for e in events if e["event"]==kind and e.get("admission_policy_defer") is True} == deferred_keys, "Unreceipted admission utility deferral"
    return verified


def summarize(folder):
    # Never silently disable evidence validation via Python -O / PYTHONOPTIMIZE.
    if not __debug__:
        raise RuntimeError("Evidence replay requires assertions; remove Python -O / PYTHONOPTIMIZE")
    m = json.loads((folder / "manifest.json").read_text())
    config = m["configuration"]
    records = json.loads((folder / "requests.json").read_text()) if (folder / "requests.json").exists() else []
    for name, expected in m["source_sha256"].items():
        assert hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() == expected, (
            "Run source hash mismatch"
        )
    path = folder / "allocator_events.jsonl"
    events = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    capacity = m.get("physical_capacity_pages")
    if m["status"] == "PASS" or (config.get("audit") and events):
        assert type(capacity) is int and capacity >= 0, "Evidence replay requires measured physical capacity"
    if config.get("active_drain_admission",False) and (events or m["status"]=="PASS"):
        contract=m["active_admission_contract"];universe_path=folder/"finite_request_universe.json"
        assert hashlib.sha256(universe_path.read_bytes()).hexdigest()==contract["universe_sha256"], "Finite input universe bytes changed"
        config=dict(config,_active_input_universe=json.loads(universe_path.read_text()),_active_input_universe_sha=contract["universe_sha256"])
    event_index=_EventIndex(events)
    active_admission_observation = _verify_cohort_certified_batches(events, config, capacity)
    if config.get("active_drain_admission",False) and m["status"]=="PASS":
        assert active_admission_observation["exact_work_evidence"], "UNKNOWN cancelled spans cannot pass exact-work/performance evidence"
    ordinary_observation = _verify_ordinary_priority(events, config, capacity)
    head_funding_verified = _verify_head_funded(events, config, capacity,_event_index=event_index)
    admission_deferrals = _verify_admission_policy(events, config, m.get("request_caps"),_event_index=event_index)
    proposal_manifest_path = folder / "proposals.json"
    proposal_manifest = json.loads(proposal_manifest_path.read_text()) if proposal_manifest_path.exists() else []
    if proposal_manifest:
        assert isinstance(proposal_manifest, list) and all(isinstance(row, dict) for row in proposal_manifest)
        proposal_ids = [row.get("request_id") for row in proposal_manifest]
        assert all(isinstance(rid, str) and rid for rid in proposal_ids) and len(set(proposal_ids)) == len(proposal_ids)
        for row in proposal_manifest:
            assert isinstance(row.get("action_id"), str) and row["action_id"]
            assert isinstance(row.get("worker_id"), str) and isinstance(row.get("pool_id"), str)
            assert isinstance(row.get("state_version"), int) and row["state_version"] >= 0
            assert row["hard_max_output_tokens"] >= 0 and row["hard_need_pages"] >= 0
            assert row["soft_reserve_tokens"] <= row["hard_max_output_tokens"]
            assert row["route"]["device_id"] == row["worker_id"] == "gpu0"
            assert tuple(row["placement"]["devices"]) == ("gpu0",)
            assert row.get("predictor_checkpoint_sha256")
            assert row.get("predictor_split_sha256")
    # Core adds random suffixes; external IDs start w<wave>-r<index>.
    work = [
        e
        for e in events
        if e.get("request_id", "").startswith("w") and not e.get("request_id", "").startswith("warmup")
    ]
    alloc = [e for e in work if e["event"] == "allocation"]
    release = [e for e in work if e["event"] == "release"]
    proposals = [e for e in work if e["event"] == "proposal"]
    proposal_bindings = [e for e in work if e["event"] == "proposal_binding"]
    causal_events = [e for e in work if e["event"].startswith("causal_")]
    incremental = [e for e in work if e["event"] == "incremental_proposal"]
    latest_proposal = {}
    successful_reuse = successful_zero_owner_reuse = 0
    for event in work:
        if event["event"] == "proposal":
            latest_proposal[event["request_id"]] = event
        elif event["event"] == "allocation":
            proposal = latest_proposal.pop(event["request_id"])
            assert proposal["step"] == event["step"], "Allocation/proposal trace mismatch"
            successful_reuse += proposal["reused_blocks"] > 0
            successful_zero_owner_reuse += proposal["touched_free"] > 0
    growth_fast = zero_fast = 0
    pending_incremental = None
    for event in work:
        if event["event"] == "incremental_proposal":
            pending_incremental = event
        elif event["event"] == "proposal" and pending_incremental is not None:
            assert (event["step"], event["request_id"]) == (
                pending_incremental["step"],
                pending_incremental["request_id"],
            )
            if pending_incremental["fixed_accepted"]:
                growth_fast += event["new_pages"] > 0
                zero_fast += event["new_pages"] == 0
            pending_incremental = None
    snapshots = [
        e
        for e in events
        if e["event"] == "snapshot"
        and e["cohort"]
        and all(r.startswith("w") and not r.startswith("warmup") for r in e["cohort"])
    ]
    witnesses = []
    subset = shared = useful = replayed = bridge_count = closure_gain_count = phase_count = prefill_count = 0
    for e in snapshots:
        rank = {rid: i for i, rid in enumerate(e["cohort"])}
        groups = tuple(Group(str(g["block"]), 0, 1, frozenset(rank[r] for r in g["owners"])) for g in e["groups"])
        s = State((e["free"],), tuple((v,) for v in e["hard_need"]), groups)
        boundary = boundary_report(s)
        bridge_count += boundary["bridge_holds"] is True
        closure_gain_count += boundary["closure_gain"]
        phases = e.get("request_phases")
        if phases is not None:
            assert len(phases) == s.n and {r["request_id"] for r in phases} == set(e["cohort"]), "Phase/cohort mismatch"
            phase_count += 1
            prefill_count += all(r["prefill_complete"] for r in phases)
        assert s.capacity == (m["physical_capacity_pages"],), "Replay capacity mismatch"
        pinned = State(s.free, s.need, tuple(Group(g.key, 0, 1, g.owners, len(g.owners) > 1) for g in groups))
        full = len(indexed_closure(s)) == s.n
        logged_certificate = len(indexed_closure(pinned)) == s.n if config["method"] == "exclusive_closure" else full
        assert logged_certificate == e["certificate"], "Replay differs from logged certificate"
        gain = full and len(indexed_closure(pinned)) != s.n
        if gain and len(witnesses) < 3:
            witnesses.append(
                {
                    "run_id": folder.name,
                    "step": e["step"],
                    "free": e["free"],
                    "cohort": e["cohort"],
                    "hard_need": e["hard_need"],
                    "groups": e["groups"],
                    "completion_order": [e["cohort"][i] for i in indexed_closure(s)],
                    "comparison": "same state, but all currently shared groups pinned forever",
                }
            )
        useful += gain
        shared += any(len(g.owners) > 1 for g in groups)
        subset += any(1 < len(g.owners) < s.n for g in groups)
        replayed += 1
    workload_steps = {e["step"] for e in work + snapshots}
    timing = [e["elapsed_ns"] for e in events if e["event"] == "schedule_timing" and e["step"] in workload_steps]
    if m["status"] == "PASS":
        assert len(records) == config["count"] * config["waves"]
        assert all(r["terminal"] in ("completed", "cancelled") for r in records)
        assert all(len(r["tokens"]) <= r["cap"] for r in records)
        assert all(e.get("projection_verified") for e in alloc), "Unverified successful allocation"
        if config["method"] != "native_plain":
            assert release and release[-1]["all_pages_reclaimable"], "Final allocator leak"
        if config["method"] in (
            "exclusive_closure",
            "shared_closure",
            "shared_incremental",
            "prc",
            "hard_reserve",
        ):
            assert all(e["certificate"] for e in alloc + release), "Controlled certificate failure"
            assert not any(e["event"] == "preemption" for e in work), "Controlled run preempted"
            if m.get("proposal_count", 0):
                assert len(proposal_manifest) == m["proposal_count"], "proposal manifest count mismatch"
                if config["method"] == "prc":
                    expected_solver = "PATCH-Exact" if config["proposal_mode"] == "exact" else "PATCH-Learned-Neural"
                    assert all(
                        row["placement"]["solver"].startswith(expected_solver)
                        or (config["proposal_mode"] == "learned" and row["placement"]["solver"] == "PATCH-Learned+exact-fallback")
                        for row in proposal_manifest
                    ), "proposal placement solver does not match configured path"
                request_ids = {r["request_id"] for r in records if r["request_id"].startswith("w")}
                # vLLM may append an internal suffix. The adapter records the
                # exact external identity; never infer it by splitting on '-'.
                bound_ids = {e.get("external_request_id", e["request_id"]) for e in proposal_bindings}
                assert bound_ids <= set(proposal_ids), "Proposal binding references an unknown external request"
                lifecycle_path = folder / "lifecycle.json"
                lifecycle_rows = json.loads(lifecycle_path.read_text()) if lifecycle_path.exists() else []
                waiting_aborts = {a["request_id"] for a in lifecycle_rows if a["event"] == "waiting_abort_twice"}
                unexecuted_cancelled = set()
                for r in records:
                    rid = r["request_id"]
                    if rid not in waiting_aborts or r["terminal"] != "cancelled" or r["tokens"]:
                        continue
                    matching = [e for e in release if e["request_id"] == rid or e["request_id"].startswith(rid + "-")]
                    if (len(matching) == 1 and matching[0]["logical_pages"] == 0
                            and "ABORT" in matching[0]["request_status"]
                            and not any(e["request_id"] == matching[0]["request_id"] for e in alloc)):
                        unexecuted_cancelled.add(rid)
                assert request_ids - unexecuted_cancelled <= bound_ids, "Some controlled requests bypassed proposal binding"
                assert causal_events, "Controlled run has no CausalCommit events"
        assert all(e["equivalent"] for e in incremental if e.get("oracle_checked", True))
        bindings_by_step = {(e["step"], e["request_id"]): e for e in proposal_bindings}
        proposals_by_step = {(e["step"], e["request_id"]): e for e in proposals}
        allocated_steps = {(e["step"], e["request_id"]) for e in alloc}
        frozen_verified = _verified_frozen_deferrals(events, config, m.get("physical_capacity_pages"),_event_index=event_index)
        for e in incremental:
            if e.get("oracle_checked", True) or e["fixed_accepted"]:
                continue
            key = (e["step"], e["request_id"])
            binding = bindings_by_step.get(key, {})
            proposal = proposals_by_step.get(key, {})
            assert id(e) in frozen_verified or id(e) in head_funding_verified or (e.get("statistical_defer") is True
                    and binding.get("proposal_allowed") is False
                    and proposal.get("allow") is False and key not in allocated_steps), (
                "Unchecked rejection lacks a verified proposal deferral")
        assert all(e.get("oracle_checked", True) or e.get("equivalent") is None for e in incremental), (
            "Unchecked oracle marked equivalent"
        )
        if config["cancellation"]:
            lifecycle = json.loads((folder / "lifecycle.json").read_text())
            for action in lifecycle:
                if action["event"] not in ("waiting_abort_twice", "running_abort_twice"):
                    continue
                matching = [
                    e
                    for e in release
                    if e["request_id"].startswith(action["request_id"] + "-") and "ABORT" in e["request_status"]
                ]
                assert len(matching) == 1, "Cancellation missing or released twice"
                if action["event"] == "waiting_abort_twice":
                    assert matching[0]["logical_pages"] == 0
                else:
                    assert matching[0]["logical_pages"] > 0
    execution_path = folder / "execution_tokens.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    if m.get("execution_measurement") and m["status"] == "PASS":
        assert execution is not None, "Missing confirmed execution measurement"
        assert execution["pending_batches"] == 0, "Unconfirmed execution batches remain"
        assert execution["executed_tokens"] == sum(r["executed_tokens"] for r in execution["requests"].values())
        assert execution["reexecuted_tokens"] == sum(r["reexecuted_tokens"] for r in execution["requests"].values())
        assert all(0 <= r["reexecuted_tokens"] <= r["executed_tokens"] for r in execution["requests"].values())
        for record in records:
            if record["terminal"] == "completed":
                rid = record["request_id"]
                matching = [r for key, r in execution["requests"].items() if key == rid or key.startswith(rid + "-")]
                assert len(matching) == 1 and matching[0]["executed_tokens"] > 0, "Completed request missing execution evidence"
    result = {
        "run_id": folder.name,
        "status": m["status"],
        "case": config.get("case"),
        "method": config["method"],
        "execution_measurement": execution,
        "replicate": config["replicate"],
        "config": config,
        "prompt_token_sha256": m.get("prompt_token_sha256"),
        "caps_sha256": hashlib.sha256(json.dumps(m.get("request_caps")).encode()).hexdigest(),
        "completed": m.get("completed"),
        "cancelled": m.get("cancelled"),
        "output_tokens": m.get("output_tokens"),
        "measured_seconds": m.get("measured_seconds"),
        "slo_completed": m.get("slo_completed"),
        "ttft_p95_s": percentile([r.get("ttft_s") for r in records if r["terminal"] == "completed"], 0.95),
        "mean_tpot_p95_s": percentile([r.get("mean_tpot_s") for r in records if r["terminal"] == "completed"], 0.95),
        "submission_lag_p95_s": percentile(
            [r["submitted_at_s"] - r["offered_at_s"] for r in records if r["submitted_at_s"] is not None], 0.95
        ),
        "cap_terminated": sum(r.get("finish_reason") == "length" for r in records),
        "event_counts": dict(Counter(e["event"] for e in work)),
        "admission_policy_deferrals_verified": admission_deferrals,
        "head_funding_deferrals_verified": len(head_funding_verified),
        "admission_policy": config.get("admission_policy", "unrestricted"),
        "proposal_bindings": len(proposal_bindings) if path.exists() else None,
        "verified_frozen_witness_deferrals": len(frozen_verified) if m["status"] == "PASS" else None,
        "proposal_manifest_count": len(proposal_manifest) if proposal_manifest_path.exists() else None,
        "proposal_learned_exact_fallbacks": sum(
            bool(row.get("placement", {}).get("used_exact_fallback")) for row in proposal_manifest
        ) if proposal_manifest_path.exists() else None,
        "causal_authorizations": sum(e["event"] == "causal_authorize" for e in work) if path.exists() else None,
        "causal_executes": sum(e["event"] == "causal_execute" for e in work) if path.exists() else None,
        "causal_acks": sum(e["event"] == "causal_acknowledge" for e in work) if path.exists() else None,
        "causal_refunds": sum(e["event"] == "causal_refund" for e in work) if path.exists() else None,
        "causal_finishes": sum(e["event"] == "causal_finish" for e in work) if path.exists() else None,
        "snapshots_replayed": replayed if path.exists() else None,
        "shared_snapshots": shared if path.exists() else None,
        "proper_subset_shared_snapshots": subset if path.exists() else None,
        "ownership_useful_snapshots": useful if path.exists() else None,
        "ownership_closure_gain_snapshots": closure_gain_count if path.exists() else None,
        "bridge_condition_snapshots": bridge_count if path.exists() else None,
        "phase_observed_snapshots": phase_count if path.exists() else None,
        "all_prefill_complete_snapshots": prefill_count if phase_count else None,
        "allocation_checks": len(alloc) if path.exists() else None,
        "release_checks": len(release) if path.exists() else None,
        "preemptions": sum(e["event"] == "preemption" for e in work) if path.exists() else None,
        "discarded_computed_tokens": sum(e["discarded_computed_tokens"] for e in work if e["event"] == "preemption")
        if path.exists()
        else None,
        "native_failed_allocation_rollbacks": sum(
            e["event"] == "native_allocation_refusal" and e.get("rollback_audit", False) for e in work
        )
        if path.exists()
        else None,
        "waiting_cancellations_verified": sum(
            "ABORT" in e["request_status"] and e["logical_pages"] == 0 for e in release
        )
        if path.exists()
        else None,
        "resident_cancellations_verified": sum(
            "ABORT" in e["request_status"] and e["logical_pages"] > 0 for e in release
        )
        if path.exists()
        else None,
        "prefix_reuse_proposals": sum(e["reused_blocks"] > 0 for e in proposals) if path.exists() else None,
        "zero_owner_cache_touch_proposals": sum(e["touched_free"] > 0 for e in proposals) if path.exists() else None,
        "successful_prefix_reuse_allocations": successful_reuse if path.exists() else None,
        "successful_zero_owner_cache_reuse_allocations": successful_zero_owner_reuse if path.exists() else None,
        "incremental_oracle_comparisons": sum(e.get("oracle_checked", True) for e in incremental)
        if path.exists()
        else None,
        "drain_episodes": sum(e["event"] == "drain_start" for e in events) if path.exists() else None,
        "incremental_fast_accepts": sum(e["fixed_accepted"] for e in incremental) if path.exists() else None,
        "incremental_page_growth_fast_accepts": growth_fast if path.exists() else None,
        "incremental_zero_growth_fast_accepts": zero_fast if path.exists() else None,
        "incremental_complete_fallbacks": sum(e["complete_fallback"] for e in incremental) if path.exists() else None,
        "incremental_fallback_rescues": sum(e["complete_fallback"] and e["full_accepted"] for e in incremental)
        if path.exists()
        else None,
        "incremental_structural_rebuild_proposals": sum(e["structural_rebuild"] for e in incremental)
        if path.exists()
        else None,
        "incremental_commit_checks": sum(e["event"] == "incremental_commit" for e in work) if path.exists() else None,
        "proposal_controller_s": sum(e["controller_ns"] for e in proposals) / 1e9 if path.exists() else None,
        "instrumented_schedule_s": sum(timing) / 1e9 if path.exists() else None,
        "output_tokens_sha256": hashlib.sha256(
            json.dumps([(r["request_id"], r["tokens"]) for r in records]).encode()
        ).hexdigest(),
        "error": m.get("error"),
    }
    verdict_path = folder / "expected_failure_verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text())
        result["expected_failure_verdict"] = verdict
        if verdict["passed"]:
            assert m["status"] == "FAIL"
            result["status"] = "EXPECTED_REJECTION_PASS"
    if config.get("ordinary_witness_priority",False):
        result["ordinary_witness_priority_observation"] = ordinary_observation
    if config.get("active_drain_admission",False):
        result["config"]={key:value for key,value in config.items() if not key.startswith("_active_input_")}
        result["active_drain_admission_observation"]=active_admission_observation
        result["exact_work_evidence"] = active_admission_observation["exact_work_evidence"]
        result["performance_evidence_eligible"] = m["status"]=="PASS" and active_admission_observation["exact_work_evidence"]
        result["actual_work_status"] = "EXACT_CONFIRMED" if active_admission_observation["exact_work_evidence"] else "UNKNOWN_CANCELLED_SPAN"
    return result, witnesses, records


def paired_outputs(rows, records):
    pairs = []
    by_case = defaultdict(list)
    for row in rows:
        if row["status"] == "PASS":
            by_case[(row["case"], row["replicate"])].append(row)
    for (case, replicate), group in by_case.items():
        assert len({r["prompt_token_sha256"] for r in group}) == 1, f"Unpaired inputs: {case}"
        assert len({r["caps_sha256"] for r in group}) == 1, f"Unpaired caps: {case}"
        plain = next((r for r in group if r["method"] == "native_plain"), None)
        if plain:
            reference = {r["request_id"]: r["tokens"] for r in records[plain["run_id"]]}
            for row in group:
                requests = records[row["run_id"]]
                pairs.append(
                    {
                        "case": case,
                        "replicate": replicate,
                        "run_id": row["run_id"],
                        "reference": plain["run_id"],
                        "exact_output_matches": sum(r["tokens"] == reference[r["request_id"]] for r in requests),
                        "output_length_matches": sum(len(r["tokens"]) == len(reference[r["request_id"]]) for r in requests),
                        "output_length_delta_tokens": sum(len(r["tokens"])-len(reference[r["request_id"]]) for r in requests),
                        "requests": len(requests),
                    }
                )
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    witnesses = []
    records = {}
    for folder in sorted(args.results.iterdir()):
        if folder.is_dir() and (folder / "manifest.json").exists():
            status = json.loads((folder / "manifest.json").read_text())["status"]
            if status not in ("PASS", "FAIL"):
                continue
            row, examples, requests = summarize(folder)
            rows.append(row)
            witnesses += examples
            records[folder.name] = requests
    pairs = paired_outputs(rows, records)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
    (args.output / "topology_witnesses.json").write_text(json.dumps(witnesses, indent=2))
    (args.output / "paired_outputs.json").write_text(json.dumps(pairs, indent=2))
    print(
        json.dumps(
            {
                "runs": len(rows),
                "statuses": dict(Counter(r["status"] for r in rows)),
                "completed": sum(r["completed"] or 0 for r in rows),
                "cancelled": sum(r["cancelled"] or 0 for r in rows),
                "allocation_checks": sum(r["allocation_checks"] or 0 for r in rows),
                "incremental_comparisons": sum(r["incremental_oracle_comparisons"] or 0 for r in rows),
                "ownership_useful_snapshots": sum(r["ownership_useful_snapshots"] or 0 for r in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
