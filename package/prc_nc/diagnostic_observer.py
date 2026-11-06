"""W37 optional observation only. No decision, argument, return or CUDA-sync edits.

Profiled runs are diagnostic, never efficacy timings. Engine-step wall time is
NOT GPU kernel time. Scheduler cProfile cumulative times must not be summed.
"""
import atexit,cProfile,functools,json,os,pstats,time
from pathlib import Path

def dump(name,value):
 p=Path(os.environ['PRC_RUN_DIR'])/name
 p.write_text(json.dumps(value,indent=2)+'\n')

def install_scheduler(obj):
 if os.environ.get('W37_OBSERVE')!='1':return
 obj._w37_batches=[];obj._w37_preemptions=[];obj._w37_profile=cProfile.Profile()if os.environ.get('W37_PROFILE')=='1'else None
 obj._w37_scope={}
 cls=type(obj)
 if not cls.__dict__.get('_w37_wrapped',False):
  original_schedule=cls.schedule;original_update=cls.update_from_output;original_preempt=cls._preempt_request;original_reset=cls.reset_prefix_cache
  @functools.wraps(original_schedule)
  def schedule(self,*args,**kw):
   active=self._execution_timing_started;p=self._w37_profile if active else None
   return p.runcall(original_schedule,self,*args,**kw)if p else original_schedule(self,*args,**kw)
  @functools.wraps(original_update)
  def update(self,output,result,*args,**kw):
   active=self._execution_timing_started;meter=self._execution_meter
   saved=self._execution_batches.get(id(output));spans=list(meter.pending[saved[1]])if saved else[]
   budget=dict(output.num_scheduled_tokens);prompt={rid:self.requests[rid].num_prompt_tokens for rid in budget}
   before={rid:(meter.executed.get(rid,0),meter.reexecuted.get(rid,0))for rid in budget}
   p=self._w37_profile if active else None
   value=p.runcall(original_update,self,output,result,*args,**kw)if p else original_update(self,output,result,*args,**kw)
   if saved:
    assert saved[1]in meter.done and saved[1]not in meter.pending,'Observer requires model-confirmed meter batch'
    self._w37_batches.append(dict(measured=active,batch=saved[1],monotonic_ns=time.monotonic_ns(),scheduled_tokens=budget,prompt_tokens=prompt,confirmed_spans=spans,returned_request_ids=list(result.req_ids),executed_delta={rid:meter.executed.get(rid,0)-before[rid][0]for rid in budget},reexecuted_delta={rid:meter.reexecuted.get(rid,0)-before[rid][1]for rid in budget}))
   return value
  @functools.wraps(original_preempt)
  def preempt(self,request,*args,**kw):
   value=original_preempt(self,request,*args,**kw)
   if self._execution_timing_started:self._w37_preemptions.append(dict(request_id=request.request_id,monotonic_ns=time.monotonic_ns()))
   return value
  @functools.wraps(original_reset)
  def reset(self,*args,**kw):
   value=original_reset(self,*args,**kw)
   if value:self._w37_dump()
   return value
  cls.schedule=schedule;cls.update_from_output=update;cls._preempt_request=preempt;cls.reset_prefix_cache=reset;cls._w37_wrapped=True
 def finish():
  report=dict(schema_version=1,pid=os.getpid(),profiled=obj._w37_profile is not None,batches=obj._w37_batches,measured_preemption_events=obj._w37_preemptions,scope='Model-confirmed scheduler batches; warmup marked separately. Preemptions count completed scheduler preemption calls, not restart/kernel events. Observation overhead included in service time.')
  if obj._w37_profile and obj._w37_profile.getstats():
   obj._w37_profile.dump_stats(str(Path(os.environ['PRC_RUN_DIR'])/'w37_scheduler_profile.pstats'))
   st=pstats.Stats(obj._w37_profile);report['profile_functions']=[dict(file=k[0],line=k[1],name=k[2],primitive_calls=v[0],total_calls=v[1],exclusive_s=v[2],cumulative_s=v[3])for k,v in st.stats.items()]
   report['profile_scope']='Only measured schedule and update_from_output calls; includes observation overhead; excludes warmup, engine/model waits outside these callbacks; cumulative times overlap.'
  dump('w37_scheduler_observation.json',report)
 obj._w37_dump=finish
 atexit.register(finish)

def install_engine(cls):
 if os.environ.get('W37_OBSERVE')!='1'or cls.__dict__.get('_w37_engine_wrapped',False):return
 original_step=cls.step;original_reset=cls.reset_prefix_cache
 state=dict(active=False,started=False,steps=[])
 @functools.wraps(original_step)
 def step(self,*args,**kw):
  active=state['active'];start=time.monotonic_ns();value=original_step(self,*args,**kw);end=time.monotonic_ns()
  if active:state['steps'].append(dict(start_monotonic_ns=start,end_monotonic_ns=end,seconds=(end-start)/1e9))
  return value
 @functools.wraps(original_reset)
 def reset(self,*args,**kw):
  value=original_reset(self,*args,**kw)
  if value:
   if not state['started']:state.update(started=True,active=True)
   else:state['active']=False
  return value
 cls.step=step;cls.reset_prefix_cache=reset;cls._w37_engine_wrapped=True
 atexit.register(lambda:dump('w37_engine_observation.json',dict(schema_version=1,steps=state['steps'],scope='Measured synchronous driver engine.step wall intervals, including IPC/model/output waits and observer overhead; NOT GPU kernel time. No CUDA synchronization introduced.')))
