"""Contingent finite-input membership accounting; never grants allocation rights."""
from pathlib import Path
import hashlib
import json
import re

SEMANTICS = 'settled_authority_witness_at_epoch_start; membership_expands_only_after_physical_ACK; epoch_head_fixed'


def token_sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens),separators=(',',':')).encode()).hexdigest()


def finite_universe(prompts,caps,*,block_size,capacity):
    if len(prompts)!=16 or len(caps)!=16:
        raise ValueError('active admission requires exactly16 fixed workload inputs')
    rows=[]
    for i,(ids,cap) in enumerate(zip(prompts,caps)):
        if (not isinstance(ids,(list,tuple)) or not ids or any(type(t)is not int or t<0 for t in ids)
                or type(cap)is not int or cap<=0
                or (len(ids)+cap+block_size-1)//block_size>capacity):
            raise ValueError('finite input must be exact and individually feasible')
        rows.append(dict(external_id=f'w0-r{i}',prompt_token_sha256=token_sha(ids),prompt_tokens=len(ids),max_tokens=cap))
    return dict(schema='finite-active-admission-inputs-v1',workload=rows,
        warmup=dict(external_id='warmup',prompt_token_sha256=token_sha(prompts[0]),prompt_tokens=len(prompts[0]),max_tokens=8),
        universe_scope='Exactly16 workload identities plus one separately bounded warmup; not streaming admission fairness')


class ActiveAdmission:
    def __init__(self,document,*,sha256):
        if document.get('schema')!='finite-active-admission-inputs-v1' or len(document.get('workload',[]))!=16:
            raise ValueError('input-bound exact16 universe required')
        rows=document['workload']+[document['warmup']]
        if [r['external_id'] for r in rows]!=[f'w0-r{i}' for i in range(16)]+['warmup']:
            raise ValueError('finite workload and separate warmup identities differ')
        if any(type(r['max_tokens'])is not int or r['max_tokens']<=0 or type(r['prompt_tokens'])is not int or r['prompt_tokens']<=0
               or re.fullmatch('[0-9a-f]{64}',r['prompt_token_sha256']) is None for r in rows):
            raise ValueError('invalid immutable input contracts')
        if rows[-1]['max_tokens']!=8:raise ValueError('warmup cap must remain8')
        self.rows={r['external_id']:dict(r) for r in rows}
        self.sha256=sha256;self.bound={};self.retired=set();self.terminal_outcomes={};self.epoch=None

    @classmethod
    def load(cls,path,expected):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('finite input universe SHA changed')
        return cls(json.loads(raw),sha256=expected)

    def _validated_identity(self,request):
        rid=request.request_id
        matches=[external for external in self.rows if rid==external or re.fullmatch(re.escape(external)+'-[0-9a-f]{8}',rid)]
        if len(matches)!=1:raise RuntimeError('request is outside exact bound finite input universe')
        external=matches[0];row=self.rows[external]
        if (request.max_tokens!=row['max_tokens'] or request.num_prompt_tokens!=row['prompt_tokens']
                or token_sha(request.prompt_token_ids)!=row['prompt_token_sha256']):
            raise RuntimeError('native request prompt/hard cap differs from finite input binding')
        if external in self.bound and self.bound[external]!=rid:raise RuntimeError('finite external identity replayed with another native UUID')
        return external

    def bind(self,request):
        external=self._validated_identity(request)
        if external in self.retired:raise RuntimeError('terminal finite input cannot be admitted again')
        self.bound[external]=request.request_id
        return external

    def prepare_terminal(self,request,*,cancelled,completed):
        """Validate a trusted native terminal; never authorize or join a member.

        A queued cancellation can precede the first allocation proposal. Bind
        its exact immutable contract before physical release, then retire only
        after the existing Authority has verified that release. Repeated same
        UUID/outcome remains a notification, not a new admission.
        """
        if type(cancelled)is not bool or type(completed)is not bool or cancelled==completed:
            raise RuntimeError('finite terminal requires exactly one trusted terminal outcome')
        external=self._validated_identity(request)
        outcome='cancelled' if cancelled else 'completed'
        previous=self.terminal_outcomes.get(external)
        if previous is not None:
            if previous!=outcome:raise RuntimeError('contradictory trusted terminal outcome')
            return dict(external_request_id=external,outcome=outcome,new_binding=False,repeated=True)
        if external in self.retired:raise RuntimeError('retired identity lacks matching trusted terminal record')
        new_binding=external not in self.bound
        if new_binding and not cancelled:raise RuntimeError('never-admitted input cannot report successful completion')
        self.bound[external]=request.request_id
        return dict(external_request_id=external,outcome=outcome,new_binding=new_binding,repeated=False)

    def begin(self,*,episode,batch,head,members,version):
        if self.epoch is not None:raise RuntimeError('previous active model epoch is not settled')
        if not members or len(set(members))!=len(members) or head!=members[0]:raise RuntimeError('invalid active epoch cut')
        if not set(members)<=set(self.bound.values()):raise RuntimeError('resident lacks finite identity binding')
        self.epoch=dict(episode=episode,batch_index=batch,head=head,initial_members=list(members),members=list(members),
            authority_version_at_cut=version,native_accumulator=None,head_allocation=None,joins=[],cancelled=set())

    def observe_accumulator(self,request,scheduled):
        e=self.epoch
        if e is None:raise RuntimeError('active proposal outside current captured epoch')
        if e['native_accumulator'] is None and request.request_id==e['head']:
            if not isinstance(scheduled,dict):raise RuntimeError('head lacks actual native accumulator')
            e['native_accumulator']=scheduled

    def may_join(self,request,scheduled):
        e=self.epoch
        return bool(e is not None and e['head_allocation'] is not None
            and scheduled is e['native_accumulator'] and scheduled.get(e['head'],0)==e['head_allocation']['tokens']>0
            and request.request_id not in e['members'])

    def allocation_acked(self,request,grant,before_ids,after_ids,authority,*,tokens,charge):
        e=self.epoch
        if e is None:raise RuntimeError('ACK outside captured epoch')
        if authority.pending is not None or authority.confirmed!=authority.virtual or tuple(after_ids)!=authority.ids:
            raise RuntimeError('membership cannot commit before physical Authority ACK')
        if grant.request_id!=request.request_id or authority._outcomes.get(grant.grant_id) is None:
            raise RuntimeError('membership lacks exact successfully executed issued grant')
        if authority.version!=grant.state_version+1 or tuple(grant.pages)!=tuple(charge):
            raise RuntimeError('ACK version or total physical charge differs')
        if type(tokens)is not int or tokens<=0:raise RuntimeError('active allocation must schedule positive native tokens')
        if request.request_id==e['head']:
            if e['head_allocation'] is not None:raise RuntimeError('head allocated twice in one native batch')
            e['head_allocation']=dict(grant_id=grant.grant_id,tokens=tokens,authority_version=authority.version)
        if request.request_id in before_ids:
            if set(after_ids)!=set(before_ids):raise RuntimeError('resident growth changed membership')
            return None
        if set(before_ids)!=set(e['members']) or set(after_ids)!=set(before_ids)|{request.request_id}:
            raise RuntimeError('NEW ACK membership delta is not exactly its candidate')
        if not self.may_join(request,e['native_accumulator']):raise RuntimeError('NEW ACK lacks successful current-head allocation/budget')
        external=self.bind(request)
        if external=='warmup':raise RuntimeError('warmup cannot join an active workload cohort')
        record=dict(episode=e['episode'],batch_index=e['batch_index'],head=e['head'],request_id=request.request_id,
            external_request_id=external,action_id=grant.action_id,grant_id=grant.grant_id,
            proposal_version=grant.state_version-1,authorized_version=grant.state_version,acked_version=authority.version,
            before_members=list(e['members']),after_members=list(e['members'])+[request.request_id],
            initial_members=list(e['initial_members']),head_allocated_tokens=e['head_allocation']['tokens'],
            head_grant_id=e['head_allocation']['grant_id'],scheduled_tokens=tokens,total_physical_charge=list(charge),
            universe_sha256=self.sha256,physical_ACK_observed=True)
        e['members'].append(request.request_id);e['joins'].append(record)
        return record

    def terminal(self,request,*,cancelled,completed=None):
        # Called only after Authority.release has verified actual last-owner release.
        completed=not cancelled if completed is None else completed
        checked=self.prepare_terminal(request,cancelled=cancelled,completed=completed)
        if checked['repeated']:return False
        external=checked['external_request_id']
        self.terminal_outcomes[external]=checked['outcome'];self.retired.add(external)
        if self.epoch is not None and cancelled and request.request_id in self.epoch['members']:
            self.epoch['cancelled'].add(request.request_id)
        return True

    def proof(self):
        e=self.epoch
        if e is None:raise RuntimeError('no captured epoch')
        return dict(initial_members=list(e['initial_members']),final_members=list(e['members']),
            join_grant_ids=[r['grant_id'] for r in e['joins']],universe_sha256=self.sha256)
