"""Optional observation only, for synchronous vLLM token-execution diagnostics."""
import atexit
import json
import os
import time
from pathlib import Path
import vllm
from vllm.v1.core.sched.scheduler import Scheduler
from prc_nc.execution_meter import ExecutionTokenMeter


class ExecutionMeasurementMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._execution_meter = None
        self._execution_schedule_ns = 0
        self._execution_update_ns = 0
        self._execution_timing_started = False
        self._execution_schedule_calls = 0
        self._execution_update_calls = 0
        if os.environ.get("PRC_MEASURE_EXECUTION", "0") == "1":
            if (vllm.__version__ != "0.29.0" or self.scheduler_config.async_scheduling
                    or self.num_lookahead_tokens or self.connector is not None
                    or self.parallel_config.tensor_parallel_size != 1
                    or self.parallel_config.pipeline_parallel_size != 1):
                raise RuntimeError("execution measurement requires audited synchronous single GPU")
            self._execution_meter = ExecutionTokenMeter()
            self._execution_batches = {}
            self._execution_serial = 0
            self._execution_path = Path(os.environ["PRC_RUN_DIR"]) / "execution_tokens.json"
            atexit.register(self._flush_execution_meter)

        from prc_nc.diagnostic_observer import install_scheduler
        install_scheduler(self)

    def _update_after_schedule(self, scheduler_output):
        if self._execution_meter is not None and scheduler_output.num_scheduled_tokens:
            batch = self._execution_serial
            self._execution_serial += 1
            spans = [(rid, self.requests[rid].num_computed_tokens, count)
                     for rid, count in scheduler_output.num_scheduled_tokens.items()]
            if os.environ.get('PRC_MEASURE_EXECUTION_PHASES','0')=='1':
                for rid,_,_ in spans:
                    self._execution_meter.register_prompt(rid,self.requests[rid].num_prompt_tokens)
            self._execution_meter.prepare(batch, spans)
            self._execution_batches[id(scheduler_output)] = (scheduler_output, batch)
        return super()._update_after_schedule(scheduler_output)

    def update_from_output(self, scheduler_output, model_runner_output):
        # vLLM reaches this boundary after GPU writes complete. Scheduling alone
        # never increments executed counts. No admission/placement decision changes.
        started = time.perf_counter_ns() if self._execution_meter is not None else 0
        result = super().update_from_output(scheduler_output, model_runner_output)
        if self._execution_meter is not None and scheduler_output.num_scheduled_tokens:
            saved, batch = self._execution_batches.pop(id(scheduler_output))
            if saved is not scheduler_output:
                raise AssertionError("execution batch identity mismatch")
            returned=set(model_runner_output.req_ids)
            spans=self._execution_meter.pending[batch]
            missing={rid for rid,_,_ in spans}-returned
            active=getattr(self,"_active_join",None) if getattr(self,"_active_drain_admission",False) else None
            if missing and active is not None and active.epoch is not None:
                trusted={rid for rid in active.epoch["cancelled"] if getattr(self,"_authority",None) is not None
                    and self._authority._terminals.get(rid)=="cancelled"}
                self._execution_meter.settle_cancelled_unobserved(batch,returned,trusted)
            else:
                self._execution_meter.confirm(batch, model_runner_output.req_ids)
            self._execution_update_ns += time.perf_counter_ns() - started
            self._execution_update_calls += 1
            if not self.has_requests():
                self._flush_execution_meter()
        return result

    def reset_prefix_cache(self, *args, **kwargs):
        result = super().reset_prefix_cache(*args, **kwargs)
        # The experiment's first successful cache reset ends warmup. Later
        # wave/final resets preserve cumulative measured-workload timing.
        if self._execution_meter is not None and result and not self._execution_timing_started:
            self._execution_schedule_ns = self._execution_update_ns = 0
            self._execution_schedule_calls = self._execution_update_calls = 0
            self._execution_timing_started = True
        self._flush_execution_meter()
        return result

    def _flush_execution_meter(self):
        if self._execution_meter is not None:
            temp = self._execution_path.with_suffix(".tmp")
            all_requests = os.environ.get('PRC_EXECUTION_ALL_REQUESTS', '0') == '1'
            report = self._execution_meter.report(request_ids=(
                self._execution_meter.executed if all_requests else None))
            report['request_selection'] = 'all_internal_including_warmup' if all_requests else 'legacy_w_prefix'
            if all_requests:
                report['scope'] = 'all confirmed internal request positions including warmup; select workload using recorded engine request IDs; excludes padding and cached-prefix reads'
            valid = self._execution_timing_started and self._execution_schedule_calls > 0 and self._execution_update_calls > 0
            report.update(timing_status="measured" if valid else "unavailable",
                          timing_boundary_started=self._execution_timing_started,
                          measured_schedule_calls=self._execution_schedule_calls if self._execution_timing_started else 0,
                          measured_update_calls=self._execution_update_calls if self._execution_timing_started else 0,
                          schedule_seconds=self._execution_schedule_ns/1e9 if valid else None,
                          update_seconds=self._execution_update_ns/1e9 if valid else None,
                          scheduler_lifecycle_seconds=(self._execution_schedule_ns+self._execution_update_ns)/1e9 if valid else None,
                          timing_scope="schedule plus nonempty execution-result processing; after first successful cache reset; excludes warmup, initialization, client submission, cache reset and log-file writes")
            temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temp.replace(self._execution_path)


class MeasuredNativeScheduler(ExecutionMeasurementMixin, Scheduler):
    """Native schedule/preemption algorithms with execution-result observation."""

    def schedule(self, *args, **kwargs):
        started = time.perf_counter_ns()
        output = super().schedule(*args, **kwargs)
        self._execution_schedule_ns += time.perf_counter_ns() - started
        self._execution_schedule_calls += 1
        return output
