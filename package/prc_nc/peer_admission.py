"""New-resident utility gate; never certifies, reserves or authorizes memory."""
from dataclasses import asdict
from .admission import AdmissionContext

def peer_admission_decision(limit,context):
    if type(limit)is not int or limit not in (0,2,4):
        raise ValueError('supported peer limits:0,2,4')
    if type(context)is not AdmissionContext:
        raise ValueError('immutable native cap context required')
    caps=tuple(cap for _,cap in context.incumbent_caps)
    maximum=max(caps,default=0)
    shorter=context.candidate_cap<maximum
    peers=sum(cap>=context.candidate_cap for cap in caps)
    allows=limit==0 or shorter or peers<limit
    return dict(limit=limit,context=asdict(context),max_incumbent_cap=maximum,
                shorter_cap=shorter,peers_at_least_candidate_cap=peers,allows=allows)


def pair_funding_exemption(gate, free_pages, incumbent_needs, candidate_upper_pages):
    """Relax a utility refusal when any two largest claims fit independently.

    The candidate's full prompt-plus-output envelope is a conservative bound;
    incumbents use the settled Authority's actual remaining page obligations.
    This never grants memory and does not replace the original certificate.
    """
    if gate['limit'] != 4 or gate['allows'] is not False:
        raise ValueError('pair funding only supplements a fixed4 utility refusal')
    if type(free_pages) is not int or free_pages < 0:
        raise ValueError('nonnegative settled free pages required')
    if type(candidate_upper_pages) is not int or candidate_upper_pages <= 0:
        raise ValueError('positive conservative candidate envelope required')
    if (type(incumbent_needs) is not tuple or
            any(type(v) is not tuple or len(v) != 2 or type(v[0]) is not str
                or type(v[1]) is not int or v[1] < 0 for v in incumbent_needs) or
            incumbent_needs != tuple(sorted(incumbent_needs)) or
            tuple(r for r, _ in incumbent_needs) != tuple(r for r, _ in gate['context']['incumbent_caps'])):
        raise ValueError('settled needs must match the native incumbent identities')
    claims = sorted([candidate_upper_pages] + [d for _, d in incumbent_needs], reverse=True)
    critical_pair_pages = sum(claims[:2])
    funded = free_pages >= critical_pair_pages
    proof = dict(free_pages=free_pages, incumbent_need_pages=incumbent_needs,
                 candidate_upper_pages=candidate_upper_pages,
                 critical_pair_pages=critical_pair_pages, pair_funded=funded,
                 semantics='two_largest_remaining_claims_no_release_or_sharing_credit')
    return dict(gate, base_allows=False, capacity_exemption=funded,
                pair_funding=proof, allows=funded)
