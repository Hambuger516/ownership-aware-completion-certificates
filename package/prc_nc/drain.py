"""T8's explicit finite-start service obligation, separate from certificate safety."""

from .model import integer


def terminal_flags(status):
    """Only the reviewed vLLM terminal statuses attest completed service."""
    # RequestStatus is IntEnum: on Python3.12+ str(member) is numeric, whereas
    # member.name retains the reviewed terminal reason. String fixtures and
    # explicit serialized enum names remain supported; bare integers do not.
    name = getattr(status, "name", None)
    text = name if isinstance(name, str) else status if isinstance(status, str) else ""
    text = text.strip().upper().rsplit(".", 1)[-1]
    cancelled = text in ("FINISHED_ABORTED", "FINISHED_CANCELLED", "FINISHED_CANCELED", "CANCELLED", "CANCELED")
    completed = text in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
    return cancelled, completed


class DrainPolicy:
    def __init__(self, after_steps=64, *, scope="cohort", batching="serial", active_admission=False):
        self.after_steps = integer(after_steps, "drain trigger steps", 1)
        if scope not in ("cohort", "head", "first_completion"):
            raise ValueError("invalid drain scope")
        if batching not in ("serial", "witness_preserving", "cohort_certified", "head_funded"):
            raise ValueError("invalid drain batching")
        if batching == "witness_preserving" and scope != "cohort":
            raise ValueError("witness-preserving batching requires cohort scope")
        if batching == "cohort_certified" and scope not in ("cohort", "first_completion"):
            raise ValueError("cohort-certified batching requires cohort or first-completion scope")
        if scope == "first_completion" and batching != "cohort_certified":
            raise ValueError("first-completion scope requires cohort-certified batching")
        if batching == "head_funded" and scope != "head":
            raise ValueError("head-funded batching requires head scope")
        if type(active_admission) is not bool or (active_admission and (scope,batching)!=("cohort","cohort_certified")):
            raise ValueError("active admission requires cohort_certified whole-cohort mode")
        self.active_admission = active_admission
        self._active_epoch_head = None
        self.scope = scope
        self.batching = batching
        self.elapsed = 0
        self.order = ()
        self.active = False
        self.episodes = 0
        self.head_completions = 0
        self.member_completions = 0
        self.completed_segments = 0
        self.empty_without_head_completion = 0
        self.cancelled_members = 0
        self._segment_head_completions = 0
        self._segment_member_completions = 0
        self.selected_head = None
        self._first_completion_epoch_head = None
        self.replacement_pending = False
        self.last_selection = None

    def observe(self, ids, witness):
        ids = tuple(ids)
        witness = tuple(witness)
        self.last_selection = None
        if self.active:
            if not set(ids) <= set(self.order):
                raise RuntimeError("new protected request appeared during frozen drain")
            if (self.scope == "head" or self.batching in ("witness_preserving", "cohort_certified")) and set(ids) != set(self.order):
                raise RuntimeError("head-drain member disappeared without a confirmed terminal callback")
            if self.batching == "head_funded" and ids:
                if self.selected_head is None:
                    if not self.replacement_pending:
                        raise RuntimeError("selected head missing without confirmed cancellation")
                    if len(witness) != len(ids) or set(witness) != set(range(len(ids))):
                        raise RuntimeError("replacement requires a complete current Authority witness")
                    self.selected_head = ids[witness[0]]
                    self.order = (self.selected_head,) + tuple(r for r in self.order if r != self.selected_head)
                    self.replacement_pending = False
                    self.last_selection = "selected_head_cancelled"
                elif self.selected_head != self.order[0]:
                    raise RuntimeError("surviving selected head cannot change")
            elif self.batching == "cohort_certified" and ids:
                if len(witness) != len(ids) or set(witness) != set(range(len(ids))):
                    raise RuntimeError("cannot refresh cohort without a complete current witness")
                # The resident membership is frozen, but this epoch may use a
                # different legal Authority witness after confirmed prior work.
                self.order = tuple(ids[i] for i in witness)
                if self.active_admission:self._active_epoch_head=self.order[0]
                if self.scope == "first_completion":
                    self._first_completion_epoch_head = self.order[0]
            else:
                self.order = tuple(r for r in self.order if r in ids)
            if not self.order:
                self._end("cohort_empty_without_head_completion")
            return False
        if not ids:
            self.elapsed = 0
            return False
        self.elapsed += 1
        if self.elapsed < self.after_steps:
            return False
        if len(witness) != len(ids) or set(witness) != set(range(len(ids))):
            raise RuntimeError("cannot drain without a complete witness")
        self.order = tuple(ids[i] for i in witness)
        if self.active_admission:self._active_epoch_head=self.order[0]
        if self.scope == "first_completion":
            self._first_completion_epoch_head = self.order[0]
        if self.batching == "head_funded":
            self.selected_head = self.order[0]
            self.replacement_pending = False
            self.last_selection = "segment_start"
        self.active = True
        self.episodes += 1
        self._segment_head_completions = 0
        self._segment_member_completions = 0
        return True

    def allows(self, request_id):
        if self.active and self.batching == "head_funded":
            return self.selected_head is not None and request_id in self.order
        return not self.active or bool(self.order and (
            request_id in self.order if self.batching in ("witness_preserving", "cohort_certified")
            else request_id == self.order[0]))

    def _end(self, reason, request_id=None):
        event = dict(scope=self.scope, batching=self.batching, episode=self.episodes, reason=reason,
                     completed_request_id=request_id,
                     successful_head_completions=self._segment_head_completions,
                     successful_member_completions=self._segment_member_completions,
                     remaining_frozen_requests=list(self.order))
        if (self._segment_member_completions if self.scope == "first_completion" else self._segment_head_completions):
            self.completed_segments += 1
        else:
            self.empty_without_head_completion += 1
        self.active = False
        self.selected_head = None
        self._first_completion_epoch_head = None
        self.replacement_pending = False
        self.last_selection = None
        self.order = ()
        self.elapsed = 0
        self._segment_head_completions = 0
        self._segment_member_completions = 0
        return event

    def finished(self, request_id, *, cancelled=False, completed=None):
        """Trusted physical terminal callback, never a cancellation intent.

        Direct callers without a reason retain the original trusted-completion
        API. The engine adapter passes explicit completion/cancellation flags.
        Head mode ends only after its selected head actually completes; a
        cancelled head is removed and the next frozen head must make progress.
        """
        if type(cancelled) is not bool or (completed is not None and type(completed) is not bool):
            raise ValueError("terminal flags must be bool")
        completed = not cancelled if completed is None else completed
        if cancelled and completed:
            raise ValueError("a cancellation is not a successful completion")
        if not self.active or request_id not in self.order:
            return None
        if (self.scope == "head" or self.batching in ("witness_preserving", "cohort_certified")) and not (cancelled or completed):
            raise RuntimeError("head drain requires a confirmed completion or cancellation")
        # The first-completion winner may follow a head cancellation in the
        # same model callback. Removing that head must not retrospectively turn
        # a scheduled follower into this epoch's head.
        was_head = request_id == (self._first_completion_epoch_head if self.scope == "first_completion"
            else self._active_epoch_head if self.active_admission else self.selected_head if self.batching == "head_funded" else self.order[0])
        self.order = tuple(r for r in self.order if r != request_id)
        if cancelled:
            self.cancelled_members += 1
            if was_head and self.batching == "head_funded":
                self.selected_head = None
                self.replacement_pending = bool(self.order)
        elif completed:
            self.member_completions += 1
            self._segment_member_completions += 1
            if was_head:
                self.head_completions += 1
                self._segment_head_completions += 1
                if self.scope == "head":
                    return self._end("selected_head_completed", request_id)
            if self.scope == "first_completion":
                return self._end("first_member_completed", request_id)
        if not self.order:
            return self._end("cohort_empty_without_member_completion" if self.scope == "first_completion" else "cohort_completed" if self._segment_head_completions else "cohort_empty_without_head_completion", request_id)
        return None
