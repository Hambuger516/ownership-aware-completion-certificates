import itertools
import random
import unittest
from prc_nc.patch_exact import (PatchGraph,PatchStage,PatchLink,PatchQuery,
    PatchNormalization,PatchExactSolver,NoFeasiblePlacement,SearchLimitExceeded)
from prc_nc.patch_sequential import SequentialCost,SequentialExactSolver,evaluate_sequential


class SequentialTests(unittest.TestCase):
    def test_decode_resource_preserves_slower_objective_prefix_at_merge(self):
        graph=PatchGraph((PatchStage('a','d',0,0,.1,.09,1,1),
                         PatchStage('b','d',0,0,.2,.01,1,1),
                         PatchStage('z','d',1,1,.1,.03,1,1)),
                         (PatchLink('a','z'),PatchLink('b','z')),2,{'d':4})
        query=PatchQuery(1,.1)
        contract=SequentialCost({('a','z'):.01,('b','z'):.01},{'a':.001,'b':.001,'z':.001},.002,'test fixture')
        self.assertEqual(PatchExactSolver().solve(graph,query).path,('a','z'))
        with self.assertRaises(NoFeasiblePlacement):evaluate_sequential(graph,query,('a','z'),contract)
        result=SequentialExactSolver().solve(graph,query,contract)
        self.assertEqual(result.path,('b','z'));self.assertAlmostEqual(result.tpot_seconds,.054)
        with self.assertRaises(SearchLimitExceeded):SequentialExactSolver(max_labels=1).solve(graph,query,contract)
        with self.assertRaises(ValueError):SequentialExactSolver().solve(graph,query,SequentialCost({},contract.stage_overheads,0,'missing links'))

    def test_random_small_graphs_against_independent_cartesian_enumeration(self):
        rng=random.Random(912)
        for _ in range(100):
            stages=tuple(PatchStage(f'{level}:{option}',f'd{option}',level,level,
                rng.uniform(.01,.1),rng.uniform(.01,.06),rng.randint(1,4),rng.uniform(.1,1))
                for level in range(3) for option in range(2))
            links=tuple(PatchLink(f'{level}:{a}',f'{level+1}:{b}',rng.uniform(0,.01))
                for level in range(2) for a in range(2) for b in range(2))
            graph=PatchGraph(stages,links,3,{'d0':6,'d1':6})
            query=PatchQuery(.5,rng.uniform(.05,.22),normalization=PatchNormalization(0,1,0,10))
            transfers={(l.source,l.target):rng.uniform(0,.01) for l in links}
            overheads={s.stage_id:rng.uniform(0,.005) for s in stages}
            contract=SequentialCost(transfers,overheads,.003,'synthetic independent enumeration')
            by_id={s.stage_id:s for s in stages};by_link={(l.source,l.target):l for l in links};valid=[]
            for choices in itertools.product(range(2),repeat=3):
                path=tuple(f'{i}:{choice}' for i,choice in enumerate(choices));selected=[by_id[k] for k in path]
                memory=[sum(s.memory_pages for s in selected if s.device_id==f'd{i}') for i in range(2)]
                decode=.003+sum(s.decode_seconds+overheads[s.stage_id] for s in selected)+sum(transfers[a,b] for a,b in zip(path,path[1:]))
                if max(memory)>6 or decode>query.tpot_budget:continue
                ttft=sum(s.prefill_seconds for s in selected)+sum(by_link[a,b].transfer_seconds for a,b in zip(path,path[1:]))
                objective=.5*ttft/(1+1e-12)+.5*sum(s.gpu_cost for s in selected)/(10+1e-12)
                valid.append((objective,path,decode))
            if not valid:
                with self.assertRaises(NoFeasiblePlacement):SequentialExactSolver().solve(graph,query,contract)
            else:
                expected=min(valid);result=SequentialExactSolver().solve(graph,query,contract)
                self.assertEqual(result.path,expected[1]);self.assertAlmostEqual(result.objective,expected[0]);self.assertAlmostEqual(result.tpot_seconds,expected[2])


if __name__=='__main__':unittest.main()
