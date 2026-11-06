"""Strict new-mode membership/model receipts; no old-mode validation is relaxed."""
from collections import defaultdict
import hashlib,json
from .active_admission import ActiveAdmission,SEMANTICS
from .model import State,Group
from .algorithms import indexed_closure
from .model import margins as reference_margins


def verify_active_admission(events,config,capacity):
    if not events:return dict(joins=0,epochs=0,exact_work_evidence=True)
    assert type(capacity)is int and capacity>=0
    assert (config.get('drain_scope'),config.get('drain_batching'),config.get('drain_policy'),config.get('admission_policy'))==('cohort','cohort_certified','timer','c0_first_shorter_cap')
    universe=config['_active_input_universe'];universe_sha=config['_active_input_universe_sha']
    validated=ActiveAdmission(universe,sha256=universe_sha);contracts=validated.rows
    block_size=config.get('block_size',16)
    assert all((r['prompt_tokens']+r['max_tokens']+block_size-1)//block_size<=capacity for r in contracts.values()), 'Finite singleton input contract exceeds physical capacity'
    by_kind=defaultdict(list);by_grant=defaultdict(list)
    for i,e in enumerate(events):
        assert e.get('source_event_index')==i
        assert type(e.get('monotonic_ns'))is int and (i==0 or e['monotonic_ns']>=events[i-1]['monotonic_ns'])
        by_kind[e['event']].append((i,e))
        if e.get('grant_id'):by_grant[e['grant_id']].append((i,e))
    bindings={};external={};retired=set();terminal_ids=set();active=set();open_segment=False
    expected_cancelled_head=None
    frames={};heads={};batches={};results={};joins=defaultdict(list);snapshots={}
    current=None;head_success=member_success=0;last_terminal=None;unknown_total=0
    def transaction(grant_id,rid,before_index):
        matches=by_grant[grant_id]
        values=[]
        for kind in ('causal_authorize','causal_execute','causal_acknowledge'):
            rows=[(i,e) for i,e in matches if e['event']==kind]
            assert len(rows)==1,'Missing/duplicate actual grant transaction receipt'
            values.append(rows[0])
        (ai,a),(xi,x),(ki,k)=values
        assert ai<xi<ki<before_index and all(e['request_id']==rid for _,e in values)
        assert a['state_version']==x['state_version'] and k['state_version']==a['state_version']+1
        assert a['authorized_pages']==x['executed_pages']==k['acknowledged_pages']
        return values
    for i,e in enumerate(events):
        kind=e['event']
        if expected_cancelled_head is not None:
            assert kind not in ('causal_finish','drain_epoch_witness','drain_batch_confirmed','drain_segment_end','drain_start'), 'Missing captured-head cancellation receipt'
        if kind=='finite_input_binding':
            rid=e['request_id'];key=e['external_request_id'];assert key in contracts and key not in retired
            assert rid not in bindings and key not in external,'Duplicate/replayed finite input identity'
            row=contracts[key]
            assert e['universe_sha256']==universe_sha and e['prompt_token_sha256']==row['prompt_token_sha256']
            assert e['max_tokens']==row['max_tokens'] and e['prompt_tokens']==row['prompt_tokens']
            bindings[rid]=key;external[key]=rid
        if current in batches and current not in results and kind in ('causal_authorize','proposal','allocation','active_drain_join','drain_epoch_witness','drain_start'):
            raise AssertionError('New work/cut preceded complete prior model settlement')
        if kind=='drain_start':
            assert not open_segment and not active
            ids=e['cohort'];assert ids and len(set(ids))==len(ids) and set(ids)<=bindings.keys()
            active=set(ids);open_segment=True;head_success=member_success=0;last_terminal=None
        elif kind=='drain_epoch_witness':
            key=(e['episode'],e['batch_index']);assert key not in frames
            assert set(frames)==set(results),'Epoch before previous model confirmation'
            assert open_segment and active and set(e['epoch_order'])==active and len(e['epoch_order'])==len(active)
            assert e['head']==e['epoch_order'][0] and e['cohort_members']==sorted(active)
            assert e['pending_grant'] is False and e['prior_model_batch_pending'] is False
            assert e['order_semantics']==SEMANTICS;frames[key]=(i,e);current=key
        elif kind=='drain_head_first':
            key=(e['episode'],e['batch_index']);assert key==current and key not in heads
            assert e['order_semantics']==SEMANTICS and e['frozen_order']==frames[key][1]['epoch_order']
            assert e['head']==frames[key][1]['head'] and e['before'].count(e['head'])==1
            assert e['after']==[e['head']]+[r for r in e['before'] if r!=e['head']]
            heads[key]=(i,e)
        elif kind=='snapshot':snapshots[e['step']]=e
        elif kind=='active_drain_join':
            key=(e['episode'],e['batch_index']);assert key==current and key in heads and key not in batches
            frame=frames[key][1];rid=e['request_id'];assert rid in bindings and bindings[rid]!='warmup' and bindings[rid] not in retired
            assert e['external_request_id']==bindings[rid] and e['universe_sha256']==universe_sha
            assert e['step']==frame['step'] and e['head']==frame['head'] and e['initial_members']==frame['epoch_order']
            assert len(e['before_members'])==len(active) and set(e['before_members'])==active and rid not in active
            assert e['after_members']==e['before_members']+[rid]
            assert all(r['grant_id']!=e['grant_id'] and r['request_id']!=rid for values in joins.values() for _,r in values)
            (ai,a),(xi,x),(ki,k)=transaction(e['grant_id'],rid,i)
            assert [v-w for v,w in zip(a['confirmed_free'],a['virtual_free'])]==e['total_physical_charge']
            assert a['action_id']==e['action_id'] and e['authorized_version']==a['state_version']==e['proposal_version']+1
            assert e['acked_version']==k['state_version'] and e['total_physical_charge']==a['authorized_pages']
            assert e['physical_ACK_observed'] is True and e['head_allocated_tokens']>0 and e['scheduled_tokens']>0
            head_chain=transaction(e['head_grant_id'],frame['head'],ai)
            assert head_chain[-1][0]<ai,'NEW authorized before actual head ACK'
            post=e['physical_post'];assert hashlib.sha256(json.dumps(post,sort_keys=True,separators=(',',':')).encode()).hexdigest()==e['physical_post_sha256']
            assert set(post['cohort'])==set(e['after_members']) and len(post['cohort'])==len(e['after_members'])
            state=State(tuple(post['free']),tuple(map(tuple,post['need'])),tuple(Group(str(j),g['pool'],g['pages'],frozenset(g['owner_ranks']),g['pinned']) for j,g in enumerate(post['owner_classes'])))
            assert state.capacity==(capacity,)
            for rank,rid_in_post in enumerate(post['cohort']):
                row=contracts[bindings[rid_in_post]]
                owned=sum(g.count for g in state.groups if rank in g.owners)
                assert state.need[rank][0]+owned==(row['prompt_tokens']+row['max_tokens']+block_size-1)//block_size, 'Joined post hard envelope differs from exact input contract'
            view=State(state.free,state.need,tuple(Group(g.key,g.pool,g.count,g.owners,g.pinned or len(g.owners)>1) for g in state.groups)) if config['certificate_variant']=='common_full_c0' else state
            assert len(indexed_closure(view))==state.n,'Actual joined post has no complete active-view certificate'
            if config.get('audit'):
                cf=[(j,r) for j,r in by_kind['study_candidate_counterfactual'] if r.get('actual_grant_id')==e['grant_id']]
                assert len(cf)==1 and cf[0][0]<xi
                value=cf[0][1];assert value['request_id']==rid and value['action_id']==e['action_id'] and value['proposal_version']==e['proposal_version'] and value['actual_allow'] is True
                assert value['projected_post']==post and value['projected_post_sha256']==e['physical_post_sha256']
            allocations=[(j,r) for j,r in by_kind['allocation'] if ki<j<i and r.get('request_id')==rid]
            assert len(allocations)==1 and allocations[0][1]['projection_verified'] is True
            joins[key].append((i,e));active.add(rid)
        elif kind=='drain_batch':
            key=(e['episode'],e['batch_index']);assert key==current and key in heads and key not in batches
            frame=frames[key][1];assert e['initial_members']==e['frozen_order']==frame['epoch_order']
            expected=frame['epoch_order']+[r['request_id'] for _,r in joins[key]]
            assert e['final_members']==expected and set(expected)==active
            assert e['join_grant_ids']==[r['grant_id'] for _,r in joins[key]] and e['universe_sha256']==universe_sha
            assert e['head']==frame['head'] and e['order_semantics']==SEMANTICS
            tokens=e['scheduled_tokens'];assert tokens and set(tokens)<=active and all(type(v)is int and v>0 for v in tokens.values())
            assert tokens.get(e['head'],0)==e['head_tokens']>0
            for _,join in joins[key]:assert tokens[join['head']]==join['head_allocated_tokens'] and tokens[join['request_id']]==join['scheduled_tokens']
            batches[key]=(i,e)
        elif kind=='causal_finish':
            rid=e['request_id'];assert rid not in terminal_ids, 'Duplicate actual terminal receipt'
            terminal_ids.add(rid)
            assert e['release_observed'] is True and e['outcome'] in ('completed','cancelled')
            if rid in bindings:retired.add(bindings[rid])
            if rid in active:
                active.remove(rid);last_terminal=(i,e)
                if e['outcome']=='cancelled' and current is not None and current not in results and rid==frames[current][1]['head']:expected_cancelled_head=rid
                if e['outcome']=='completed':
                    member_success+=1;head_success+=int(current is not None and current not in results and rid==frames[current][1]['head'])
        elif kind=='drain_head_cancelled':
            assert expected_cancelled_head==e['request_id'] and current in frames and e['request_id']==frames[current][1]['head']
            assert last_terminal and last_terminal[1]['request_id']==e['request_id'] and last_terminal[1]['outcome']=='cancelled'
            assert e['next_frozen_head'] is None and e['successful_completion'] is False
            expected_cancelled_head=None
        elif kind=='drain_segment_end':
            assert open_segment and not active and e['remaining_frozen_requests']==[]
            assert e['successful_head_completions']==head_success and e['successful_member_completions']==member_success
            open_segment=False
        elif kind=='drain_batch_confirmed':
            key=(e['episode'],e['batch_index']);assert key==current and key in batches and key not in results
            budget=batches[key][1];frame=frames[key][1];assert e['order_semantics']==SEMANTICS
            for name in ('initial_members','final_members','frozen_order','head','scheduled_tokens','head_tokens','join_grant_ids','universe_sha256'):assert e[name]==budget[name]
            missing=set(budget['scheduled_tokens'])-set(e['returned_request_ids'])
            assert missing==set(e['unobserved_cancelled_request_ids'])
            trusted={r['request_id'] for j,r in by_kind['causal_finish'] if batches[key][0]<j<i and r['outcome']=='cancelled'}
            assert missing<=trusted and set(e['trusted_cancelled_ids'])<=trusted
            expected={rid:(None if rid in missing else count) for rid,count in budget['scheduled_tokens'].items()}
            assert e['confirmed_real_positions']==expected
            assert len(e['surviving_members'])==len(active) and set(e['surviving_members'])==active
            if missing:
                assert e['actual_work_status']=='UNKNOWN_CANCELLED_SPAN' and e['confirmed_new_unique_positions'] is None and e['physical_execution_observed'] is False and e['progress_credited'] is False
                unknown_total+=len(missing)
            else:
                assert e['actual_work_status']=='EXACT_CONFIRMED' and e['confirmed_new_unique_positions']==sum(expected.values())>0 and e['physical_execution_observed'] is True
                assert expected[frame['head']]>0
            results[key]=(i,e)
    assert expected_cancelled_head is None
    assert not active and not open_segment and frames.keys()==heads.keys()==batches.keys()==results.keys()
    for key,(i,frame) in frames.items():
        assert i<heads[key][0]<batches[key][0]<results[key][0]
        if config.get('audit'):
            physical=snapshots[frame['step']];ids=physical['cohort'];assert set(ids)==set(frame['epoch_order'])
            rank={rid:j for j,rid in enumerate(ids)};conservative=config['certificate_variant']=='common_full_c0'
            state=State((physical['free'],),tuple((v,) for v in physical['hard_need']),tuple(Group(str(g['block']),0,1,frozenset(rank[r] for r in g['owners']),conservative and len(g['owners'])>1) for g in physical['groups']))
            assert state.capacity==(capacity,)
            assert all(v>=0 for row in reference_margins(state,tuple(rank[r] for r in frame['epoch_order'])) for v in row)
    return dict(joins=sum(map(len,joins.values())),epochs=len(frames),unobserved_cancelled_spans=unknown_total,exact_work_evidence=not bool(unknown_total))
