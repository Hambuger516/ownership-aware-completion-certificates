"""Independent integer/cap/lifecycle replay. No serving module imports."""
import re

def native_cap(rid,trace):
 m=re.fullmatch(r'w0-r(0|[1-9][0-9]*)(?:-[0-9a-f]{8})?',rid)
 if m:
  i=int(m.group(1));assert 0<=i<len(trace);return trace[i]['max_tokens']
 assert re.fullmatch(r'warmup(?:-[0-9a-f]{8})?',rid),'Unbound native request identity'
 return 8

def audit_peer_gate(events,trace,config,stats):
 limit=config['peer_resident_limit'];assert type(limit)is int and limit in (0,2,4)
 adaptive=config.get('peer_capacity_adaptive',False);assert type(adaptive)is bool
 assert events[0].get('peer_capacity_adaptive',False)is adaptive
 free=config['blocks']-1 if adaptive else None;held={};growth={};pair_checks=pair_exemptions=0
 def upper(rid):
  m=re.fullmatch(r'w0-r(0|[1-9][0-9]*)(?:-[0-9a-f]{8})?',rid);assert m,'Pair exemption cannot inspect unknown input identity'
  r=trace[int(m.group(1))];return(min(len(r['prompt_token_ids'])+r['max_tokens'],config['max_model_len'])+config['block_size']-1)//config['block_size']
 assert events[0]['event']=='adapter'and events[0]['peer_resident_limit']==limit
 state=stats['peer_admission_stats'];assert state['limit']==limit
 residents={};pending={};version=0;checks=deferrals=new_admissions=0;proofs={}
 for e in events:
  k=e['event'];rid=e.get('request_id');v=e.get('state_version')
  if k=='peer_admission_gate':
   assert limit>0 and not pending
   checks+=1;assert e['proof_serial']==checks and e['limit']==limit
   ctx=e['context'];rid=ctx['candidate_id'];assert rid not in residents
   assert ctx['authority_version']==version and ctx['candidate_cap']==native_cap(rid,trace)
   inc=[list(x)for x in sorted(residents.items())];assert ctx['incumbent_caps']==inc
   assert e['pending_grant']is False and e['resource_states_equal']is True and e['actual_cap_source']=='native Request.max_tokens'
   caps=list(residents.values());candidate=ctx['candidate_cap'];shorter=any(c>candidate for c in caps);peers=sum(c>=candidate for c in caps)
   expected=shorter or peers<limit
   assert e.get('adaptive_enabled',False)is adaptive
   if adaptive and not expected:
    pair_checks+=1;p=e['pair_funding'];inc_need=[[r,upper(r)-held[r]]for r in sorted(residents)];u=upper(rid)
    assert all(n>=0 for _,n in inc_need)
    critical=sum(sorted([u]+[n for _,n in inc_need],reverse=True)[:2]);funded=free>=critical
    assert p['free_pages']==free and p['incumbent_need_pages']==inc_need and p['candidate_upper_pages']==u and p['critical_pair_pages']==critical
    assert p['pair_funded']is funded and p['semantics']=='two_largest_remaining_claims_no_release_or_sharing_credit'
    assert e['base_allows']is False and e['capacity_exemption']is funded
    pair_exemptions+=funded;expected=funded
   else:assert 'pair_funding'not in e and 'capacity_exemption'not in e
   assert e['max_incumbent_cap']==max(caps,default=0)and e['shorter_cap']is shorter and e['peers_at_least_candidate_cap']==peers and e['allows']is expected
   deferrals+=not expected;proofs[(rid,version)]=expected
  elif k=='causal_authorize':
   new=rid not in residents
   if new and limit:assert proofs.get((rid,version))is True,'New actual admission bypassed peer proof'
   assert v==version+1;pending[e['grant_id']]=(rid,new);version=v
   if adaptive:assert e['confirmed_free']==[free]
  elif k=='proposal'and adaptive and e['allow']:
   assert len(pending)==1;gid,=pending;assert pending[gid][0]==rid and gid not in growth
   growth[gid]=e['new_pages']+e['reused_blocks']
  elif k=='causal_execute':
   assert e['grant_id']in pending and v==version
  elif k=='causal_acknowledge':
   actual,new=pending.pop(e['grant_id']);assert actual==rid
   if new:residents[rid]=native_cap(rid,trace);new_admissions+=1
   if adaptive:
    held[rid]=held.get(rid,0)+growth.pop(e['grant_id']);free=e['confirmed_free'][0]
    assert free>=0
   assert v==version+1;version=v
  elif k=='causal_stop':
   assert e['grant_id']in pending and v>=version;version=v
  elif k=='causal_refund':
   pending.pop(e['grant_id']);assert v>=version;version=v
  elif k=='causal_finish':
   assert not pending and rid in residents and e['outcome']=='completed'
   del residents[rid];assert v==version+1;version=v
   if adaptive:held.pop(rid);free=e['confirmed_free'][0]
  elif k=='causal_fault':raise AssertionError('Authority fault retained')
 assert not residents and not pending and version==stats['authority_version']
 assert state['checks']==checks and state['deferrals']==deferrals
 if adaptive:
  assert not held and not growth and free==config['blocks']-1
  assert state['adaptive_enabled']is True and state['pair_checks']==pair_checks and state['pair_exemptions']==pair_exemptions
 if limit==0:assert checks==deferrals==0
 return dict(pass_=True,limit=limit,adaptive=adaptive,pair_checks=pair_checks,pair_exemptions=pair_exemptions,checks_including_warmup=checks,deferrals_including_warmup=deferrals,all_actual_new_admissions_replayed=new_admissions,final_resident_count=0,final_authority_version=version,scope='Native caps and lifecycle; adaptive proof independently replays free pages and resident page holdings from allocator receipts, then recomputes the two-largest-claim sum. Peer-pass grants no memory; original certificate and allocator checks remain mandatory.')
