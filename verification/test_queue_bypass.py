"""Exercise actual transfer helper and installed native FCFS queue without CUDA."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast, collections, json, textwrap, unittest
from prc_nc.configuration import validate_config

W=Path(__file__).resolve().parents[1]


def native_queue():
    tree=ast.parse((W/'native_runtime_reference/request_queue.py').read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom)and n.module=='vllm.v1.request')]
    env={'Request':object}
    exec(compile(tree,'retained_actual_native_queue','exec'),env)
    return env['FCFSRequestQueue']


def helper():
    tree=ast.parse((W/'package/prc_nc/adapters/vllm_scheduler.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='CertificateScheduler')
    f=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_peer_skip_rejected')
    env={}
    exec(compile(ast.Module(body=[f],type_ignores=[]),'actual_queue_transfer_helper','exec'),env)
    return env['_peer_skip_rejected']


class QueueBypassTests(unittest.TestCase):
    def setup_case(self):
        q=native_queue();r=NS(request_id='long',max_tokens=512,has_encoder_inputs=False,status=NS(name='WAITING'))
        short=NS(request_id='short',max_tokens=64,has_encoder_inputs=False,status=NS(name='WAITING'))
        events=[]
        obj=NS(_peer_queue_bypass=True,_peer_last_deferred_id='long',policy=NS(value='fcfs'),
               connector=None,ec_connector=None,lora_config=None,_authority=NS(pending=None,virtual=(),confirmed=(),ids=(),version=3),
               waiting=q([r,short]),skipped_waiting=q(),_peer_queue_bypasses=0,_peer_gate_checks=9,_audit=True,
               _log=lambda event,**v:events.append(dict(event=event,**v)))
        return obj,r,short,q(),events

    def test_disabled_and_nonpeer_refusals_keep_original_break(self):
        for flag,reason in [(False,'long'),(True,None),(True,'different')]:
            obj,r,s,step,ev=self.setup_case();obj._peer_queue_bypass=flag;obj._peer_last_deferred_id=reason
            self.assertFalse(helper()(obj,obj.waiting,step,r));self.assertEqual(list(obj.waiting),[r,s]);self.assertFalse(step);self.assertFalse(ev)

    def test_transfer_removes_only_head_and_exposes_short_without_authorization(self):
        obj,r,s,step,ev=self.setup_case();old_authority=vars(obj._authority).copy()
        self.assertTrue(helper()(obj,obj.waiting,step,r));self.assertEqual(list(obj.waiting),[s]);self.assertEqual(list(step),[r])
        self.assertEqual(vars(obj._authority),old_authority);self.assertEqual(obj._peer_queue_bypasses,1)
        self.assertEqual(ev[0]['queue_before'],[['long',512],['short',64]]);self.assertEqual(ev[0]['queue_after'],[['short',64]])
        self.assertEqual(ev[0]['step_skipped_after'],[['long',512]]);self.assertIsNone(obj._peer_last_deferred_id)

    def test_actual_native_double_prepend_restores_all_deferred_FIFO(self):
        obj,r,s,step,ev=self.setup_case();q=native_queue()
        longs=[NS(request_id=f'L{i}',max_tokens=512,has_encoder_inputs=False,status=NS(name='WAITING'))for i in range(7)]
        obj.skipped_waiting=q(longs[:2]);obj.waiting=q(longs[2:]+[s]);seen=[]
        while obj.waiting or obj.skipped_waiting:
            current=obj.skipped_waiting or obj.waiting;head=current.peek_request()
            if head is s:current.pop_request();seen.append('short');continue
            obj._peer_last_deferred_id=head.request_id
            self.assertTrue(helper()(obj,current,step,head));seen.append(head.request_id)
            self.assertLessEqual(len(seen),8)
        obj.skipped_waiting.prepend_requests(step)
        self.assertEqual(list(obj.skipped_waiting),longs);self.assertEqual(len(seen),8);self.assertEqual(len({id(r)for r in obj.skipped_waiting}),7)

    def test_unsupported_native_compositions_fail_before_pop(self):
        for field,value in [('policy',NS(value='priority')),('connector',object()),('ec_connector',object()),('lora_config',object())]:
            obj,r,s,step,ev=self.setup_case();setattr(obj,field,value)
            with self.assertRaises(RuntimeError):helper()(obj,obj.waiting,step,r)
            self.assertEqual(list(obj.waiting),[r,s]);self.assertFalse(step)
        obj,r,s,step,ev=self.setup_case();r.has_encoder_inputs=True
        with self.assertRaises(RuntimeError):helper()(obj,obj.waiting,step,r)
        self.assertEqual(list(obj.waiting),[r,s])

    def test_unsettled_or_resident_state_cannot_be_transferred(self):
        for update in [dict(pending=object()),dict(virtual=('different',)),dict(ids=('long',))]:
            obj,r,s,step,ev=self.setup_case()
            for k,v in update.items():setattr(obj._authority,k,v)
            with self.assertRaises(RuntimeError):helper()(obj,obj.waiting,step,r)
            self.assertEqual(list(obj.waiting),[r,s]);self.assertFalse(ev)

    def test_configuration_pins_bool_and_reviewed_limit4(self):
        template=json.loads((W/'inputs/confirm02_back.template.json').read_text())
        template.update(private_refusal_cache=False,peer_resident_limit=4)
        self.assertTrue(validate_config(template|dict(peer_queue_bypass=True))['peer_queue_bypass'])
        for changes in [dict(peer_queue_bypass=1),dict(peer_queue_bypass='1'),dict(peer_queue_bypass=True,peer_resident_limit=2),dict(peer_queue_bypass=True,peer_resident_limit=0)]:
            with self.assertRaises(ValueError):validate_config(template|changes)

    def test_actual_native_injection_keeps_guard_before_skip_and_native_restore(self):
        adapter=ast.parse((W/'package/prc_nc/adapters/vllm_scheduler.py').read_text());values={}
        for n in adapter.body:
            if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ['anchor_running','anchor_waiting','insert_running','insert_waiting']:
                values[n.targets[0].id]=ast.literal_eval(n.value)
        native=(W/'native_runtime_reference/scheduler.py').read_text();tree=ast.parse(native)
        scheduler=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Scheduler')
        schedule=next(n for n in scheduler.body if isinstance(n,ast.FunctionDef)and n.name=='schedule')
        source=textwrap.dedent(''.join(native.splitlines(keepends=True)[schedule.lineno-1:schedule.end_lineno]))
        for part in ['running','waiting']:
            anchor=values['anchor_'+part];self.assertEqual(source.count(anchor),1)
            source=source.replace(anchor,values['insert_'+part]+anchor)
        ast.parse(source)
        self.assertIn('self.skipped_waiting.prepend_requests(step_skipped_waiting)',source)
        fragment=values['insert_waiting'];self.assertLess(fragment.index('not self._prc_allow'),fragment.index('self._peer_skip_rejected'))
        self.assertIn('continue',fragment);self.assertTrue(fragment.rstrip().endswith('break'))


if __name__=='__main__':unittest.main()
