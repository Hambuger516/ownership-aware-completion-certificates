"""Count confirmed token-position execution; proposals and cache hits do not count."""

class ExecutionTokenMeter:
    def __init__(self):
        self.pending = {}
        self.done = set()
        self.intervals = {}
        self.executed = {}
        self.reexecuted = {}
        self.prompt_lengths = {}
        self.phases = {}

    def register_prompt(self, rid, length):
        if type(rid) is not str or not rid or type(length) is not int or length<=0:
            raise ValueError('valid request and positive prompt boundary required')
        if rid in self.prompt_lengths:
            if self.prompt_lengths[rid]!=length:raise ValueError('prompt boundary changed')
            return
        if rid in self.executed:raise ValueError('cannot add boundary after confirmed execution')
        self.prompt_lengths[rid]=length
        self.phases[rid]=dict(prefill_executed_tokens=0,decode_executed_tokens=0,
            prefill_reexecuted_tokens=0,decode_reexecuted_tokens=0)

    def prepare(self, batch_id, spans):
        spans = tuple(spans)
        if batch_id in self.pending or batch_id in self.done:
            raise ValueError("duplicate execution batch identity")
        ids = [r for r, _, _ in spans]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate request in execution batch")
        if any(not isinstance(r, str) or not r or type(s) is not int or s < 0
               or type(n) is not int or n <= 0 for r, s, n in spans):
            raise ValueError("invalid token execution span")
        self.pending[batch_id] = spans

    def confirm(self, batch_id, returned_request_ids):
        if batch_id in self.done:
            return False
        spans = self.pending[batch_id]
        if not {r for r, _, _ in spans} <= set(returned_request_ids):
            raise ValueError("execution result omits scheduled requests")
        for rid, start, count in spans:
            end = start + count
            intervals = self.intervals.get(rid, [])
            repeated = sum(max(0, min(end, b)-max(start, a)) for a, b in intervals)
            if rid in self.prompt_lengths:
                boundary=self.prompt_lengths[rid];prefill=max(0,min(end,boundary)-start)
                repeated_prefill=sum(max(0,min(end,b,boundary)-max(start,a)) for a,b in intervals)
                phase=self.phases[rid]
                phase['prefill_executed_tokens']+=prefill
                phase['decode_executed_tokens']+=count-prefill
                phase['prefill_reexecuted_tokens']+=repeated_prefill
                phase['decode_reexecuted_tokens']+=repeated-repeated_prefill
            merged = []
            for a, b in sorted(intervals + [(start, end)]):
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            self.intervals[rid] = merged
            self.executed[rid] = self.executed.get(rid, 0) + count
            self.reexecuted[rid] = self.reexecuted.get(rid, 0) + repeated
        del self.pending[batch_id]
        self.done.add(batch_id)
        return True

    def settle_cancelled_unobserved(self,batch_id,returned_request_ids,trusted_cancelled_ids):
        """Settle control only; an omitted cancelled span has UNKNOWN GPU work.

        Existing confirm stays unchanged. Never record an omitted span as zero
        execution or assume it was not computed. Full exact-work gates must fail.
        """
        if batch_id in self.done:raise ValueError("cancelled batch cannot settle twice")
        spans=self.pending[batch_id];returned=set(returned_request_ids)
        missing={r for r,_,_ in spans}-returned
        if not missing or not missing<=set(trusted_cancelled_ids):
            raise ValueError("missing model IDs require exact trusted cancellation")
        observed=tuple(span for span in spans if span[0] not in missing)
        self.pending[batch_id]=observed
        try:self.confirm(batch_id,returned)
        except BaseException:
            self.pending[batch_id]=spans
            raise
        unknown=[dict(request_id=r,start=start,prepared_count=count,status="UNOBSERVED_CANCELLED_SPAN",actual_work="UNKNOWN")
                 for r,start,count in spans if r in missing]
        if not hasattr(self,"unobserved_cancelled_spans"):self.unobserved_cancelled_spans=[]
        self.unobserved_cancelled_spans.append(dict(batch_id=batch_id,prepared_spans=list(spans),
            trusted_cancelled_ids=sorted(set(trusted_cancelled_ids)),unknown_spans=unknown,exact_work_evidence=False))
        return unknown

    def report(self, *, request_ids=None):
        # Legacy runs used w-prefixed IDs. New clients must explicitly select
        # their measured cohort; names such as pilot:0 are not warmup.
        selected = None if request_ids is None else frozenset(request_ids)
        if selected is not None and any(not isinstance(r, str) or not r for r in selected):
            raise ValueError("nonempty measured request identities required")
        def workload(rid):
            if selected is not None:
                return rid in selected
            return rid.startswith("w") and not rid.startswith("warmup")
        result = dict(schema_version=1, pending_batches=len(self.pending),
                    confirmed_batches=len(self.done),
                    executed_tokens=sum(n for r, n in self.executed.items() if workload(r)),
                    reexecuted_tokens=sum(n for r, n in self.reexecuted.items() if workload(r)),
                    requests={r: dict(executed_tokens=n, reexecuted_tokens=self.reexecuted[r],**self.phases.get(r,{}))
                              for r, n in self.executed.items() if workload(r)},
                    scope="positions executed again for the same internal request after an earlier confirmed execution; excludes padding, cached-prefix reads and warmup")
        if getattr(self,"unobserved_cancelled_spans",None):
            result.update(exact_work_evidence=False,actual_work_status="UNKNOWN_CANCELLED_SPAN",
                observed_executed_tokens=result["executed_tokens"],observed_reexecuted_tokens=result["reexecuted_tokens"],
                executed_tokens=None,reexecuted_tokens=None,unobserved_cancelled_spans=self.unobserved_cancelled_spans,
                unknown_work_limit="Observed counts exclude UNOBSERVED_CANCELLED_SPAN; no exact total/zero-work claim")
        return result
