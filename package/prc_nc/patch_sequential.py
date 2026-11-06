"""Versioned additive decode cost for synchronous, nonoverlapped execution.

The historical PATCH/checkpoint semantics remain unchanged. This contract needs
explicit decode transfers and non-model overhead, independently calibrated.
It predicts latency; allocator certification remains a separate authority.
"""
from dataclasses import dataclass, replace
from types import MappingProxyType
from .patch_exact import (NoFeasiblePlacement, SearchLimitExceeded, _integer,
    _number, _risk, _risk_limit, _objective, evaluate_path,
    resolve_normalization, topological_order)


@dataclass(frozen=True)
class SequentialCost:
    decode_transfers: object
    stage_overheads: object
    endpoint_seconds: float
    provenance: str
    version: str = 'sequential-decode-v1'

    def __post_init__(self):
        if self.version!='sequential-decode-v1' or type(self.provenance) is not str or not self.provenance:
            raise ValueError('explicit version and calibration provenance required')
        transfers=dict(self.decode_transfers);overheads=dict(self.stage_overheads)
        for key,value in transfers.items():
            if type(key) is not tuple or len(key)!=2 or any(type(k) is not str or not k for k in key):
                raise ValueError('directed stage-pair transfer keys required')
            transfers[key]=_number(value,'decode transfer')
        for key,value in overheads.items():
            if type(key) is not str or not key:raise ValueError('stage overhead identity required')
            overheads[key]=_number(value,'stage overhead')
        object.__setattr__(self,'decode_transfers',MappingProxyType(transfers))
        object.__setattr__(self,'stage_overheads',MappingProxyType(overheads))
        object.__setattr__(self,'endpoint_seconds',_number(self.endpoint_seconds,'endpoint overhead'))

    def validate(self,graph):
        if set(self.decode_transfers)!={(l.source,l.target) for l in graph.links}:
            raise ValueError('explicit decode calibration required for every link')
        if set(self.stage_overheads)!={s.stage_id for s in graph.stages}:
            raise ValueError('explicit overhead calibration required for every stage')


def evaluate_sequential(graph,query,path,cost):
    cost.validate(graph)
    result=evaluate_path(graph,query,path)
    stages={s.stage_id:s for s in graph.stages}
    decode=cost.endpoint_seconds
    for i,key in enumerate(result.path):
        if i:decode+=cost.decode_transfers[result.path[i-1],key]
        decode+=stages[key].decode_seconds+cost.stage_overheads[key]
    if decode>query.tpot_budget:
        raise NoFeasiblePlacement('sequential cumulative decode budget exceeded')
    return replace(result,tpot_seconds=decode,solver='validated-sequential-v1')


@dataclass(frozen=True)
class _Label:
    path: tuple
    ttft: float
    cost: float
    risk: float
    memory: tuple
    decode: float
    scalar: float


def _dominates(a,b):
    return (a.ttft<=b.ttft and a.cost<=b.cost and a.risk<=b.risk and a.decode<=b.decode
            and all(x<=y for x,y in zip(a.memory,b.memory)))


class SequentialExactSolver:
    def __init__(self,*,max_labels=None):
        if max_labels is not None:_integer(max_labels,'max_labels',1)
        self.max_labels=max_labels

    def solve(self,graph,query,cost):
        cost.validate(graph)
        normalization=resolve_normalization(graph,query)
        stages={s.stage_id:s for s in graph.stages}
        devices=tuple(sorted(graph.device_memory_pages));indices={d:i for i,d in enumerate(devices)}
        capacities=tuple(graph.device_memory_pages[d] for d in devices)
        outgoing={k:[] for k in stages};labels={k:[] for k in stages}
        for link in graph.links:outgoing[link.source].append(link)
        expanded=pruned=0
        def extend(old,stage,link=None):
            nonlocal expanded,pruned
            decode=old.decode+(cost.decode_transfers[link.source,link.target] if link else 0)
            decode+=stage.decode_seconds+cost.stage_overheads[stage.stage_id]
            risk=old.risk+_risk(stage.availability)
            memory=list(old.memory);memory[indices[stage.device_id]]+=stage.memory_pages
            if decode>query.tpot_budget or risk>_risk_limit(query) or any(v>c for v,c in zip(memory,capacities)):return
            ttft=old.ttft+(link.transfer_seconds if link else 0)+stage.prefill_seconds
            monetary=old.cost+(link.transfer_cost if link else 0)+stage.gpu_cost
            candidate=_Label(old.path+(stage.stage_id,),ttft,monetary,risk,tuple(memory),decode,
                             _objective(ttft,monetary,query,normalization))
            expanded+=1
            if self.max_labels is not None and expanded>self.max_labels:
                raise SearchLimitExceeded('sequential exact label budget reached')
            current=labels[stage.stage_id]
            if any(_dominates(old,candidate) for old in current):pruned+=1;return
            keep=[old for old in current if not _dominates(candidate,old)]
            pruned+=len(current)-len(keep);keep.append(candidate);labels[stage.stage_id]=keep
        empty=_Label((),0.,0.,0.,(0,)*len(devices),cost.endpoint_seconds,0.)
        for key in topological_order(graph):
            stage=stages[key]
            if stage.start_layer==0:extend(empty,stage)
            for label in labels[key]:
                for link in sorted(outgoing[key],key=lambda l:l.target):extend(label,stages[link.target],link)
        terminals=[label for key,values in labels.items() if stages[key].end_layer==graph.layer_count-1 for label in values]
        if not terminals:raise NoFeasiblePlacement('no sequential feasible placement')
        best=min(terminals,key=lambda l:(l.scalar,l.path))
        result=evaluate_sequential(graph,replace(query,normalization=normalization),best.path,cost)
        return replace(result,solver='PATCH-Exact-Sequential-v1',labels_expanded=expanded,labels_pruned=pruned)
