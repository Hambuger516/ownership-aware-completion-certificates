"""A deterministic utility filter, separate from completion-certificate safety."""
from dataclasses import dataclass, asdict

POLICIES=('unrestricted','c0_first_shorter_cap')


@dataclass(frozen=True)
class AdmissionContext:
    candidate_id: str
    authority_version: int
    candidate_cap: int
    incumbent_caps: tuple[tuple[str,int], ...]

    def __post_init__(self):
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ValueError('candidate identity required')
        if type(self.authority_version) is not int or self.authority_version < 0:
            raise ValueError('Authority version required, not controller version')
        if type(self.candidate_cap) is not int or self.candidate_cap <= 0:
            raise ValueError('positive actual hard output cap required')
        if type(self.incumbent_caps) is not tuple or any(type(p) is not tuple or len(p)!=2 for p in self.incumbent_caps):
            raise ValueError('immutable incumbent cap tuples required')
        if any(type(r) is not str or not r or type(c) is not int or c<=0 for r,c in self.incumbent_caps):
            raise ValueError('invalid incumbent cap')
        ids=tuple(r for r,_ in self.incumbent_caps)
        if len(set(ids))!=len(ids) or self.candidate_id in ids or self.incumbent_caps!=tuple(sorted(self.incumbent_caps)):
            raise ValueError('sorted distinct incumbent identities required')


@dataclass(frozen=True)
class AdmissionDecision:
    policy: str
    context: AdmissionContext
    c0_complete: bool
    c1_complete: bool
    max_incumbent_cap: int
    shorter_cap: bool
    policy_allows: bool

    def summary(self):return asdict(self)


class AdmissionDeferred(ValueError):
    """A legal C1 post-state deliberately deferred by the utility policy."""
    def __init__(self,decision):
        super().__init__('legal certificate; utility admission policy deferred new resident')
        self._decision=decision

    @property
    def decision(self):return self._decision


def decide(policy, context, c0_complete, c1_complete):
    if policy not in POLICIES:raise ValueError('unsupported admission policy')
    if type(context) is not AdmissionContext:raise ValueError('immutable admission context required')
    if type(c0_complete) is not bool or type(c1_complete) is not bool:
        raise ValueError('complete-state certificate decisions must be bool')
    maximum=max((cap for _,cap in context.incumbent_caps),default=0)
    shorter=context.candidate_cap<maximum
    allowed=policy=='unrestricted' or c0_complete or (c1_complete and shorter)
    return AdmissionDecision(policy,context,bool(c0_complete),bool(c1_complete),maximum,shorter,bool(allowed))
