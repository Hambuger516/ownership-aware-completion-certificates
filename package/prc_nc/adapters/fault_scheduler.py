"""Isolated negative tests. Never used by normal runs or an existing service."""

import os
from prc_nc.adapters.vllm_scheduler import CertificateScheduler, Scheduler


class FaultScheduler(CertificateScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fault = os.environ["PRC_TEST_FAULT"]
        self._fault_armed = True

    def _snapshot(self, extra=None):
        if self._fault == "unmodeled_ref" and self._fault_armed:
            blocks = next((b for b in self._pm.req_to_blocks.values() if b), None)
            if blocks:
                self._fault_armed = False
                self._log("fault_injection", kind=self._fault, block_id=blocks[0].block_id)
                # Add a reference absent from the request-owner mapping. The
                # allocator audit must stop before this scheduler dispatches GPU
                # work. The disposable engine process then exits and frees GPU.
                blocks[0].ref_cnt += 1
        return super()._snapshot(extra)

    def schedule(self, *args, **kwargs):
        if self._fault == "bypass_guard":
            self._log("fault_injection", kind=self._fault)
            return Scheduler.schedule(self, *args, **kwargs)
        return super().schedule(*args, **kwargs)
