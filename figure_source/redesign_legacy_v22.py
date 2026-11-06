"""Redraw seven legacy/supporting figures using archived measurements only.
Run with Python 3.12 or later; no GPU, no data edits.
All size arguments are physical inches; SVG keeps editable text.
"""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

W=Path(__file__).resolve().parents[1]
D=W/'source_data'; O=D/'v22_figures/legacy'; F=W/'figs'
SINGLE=3.16697799916978; DOUBLE=6.500000276740003
for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(p)
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
C={'prc':'#285F8C','native_plain':'#28776F','hard_reserve':'#687E88','prc_original':'#A37232','prc_tuned':'#285F8C','green':'#28776F','gold':'#A37232'}
METHODS=['prc','native_plain','hard_reserve']; LABELS=['PRC','Native','Hard']; MARKS=['o','s','^']
CASES=['EFFECT_dolly_pressure','EFFECT_dolly_roomy','EFFECT_hetero_forced']; CL=['Pressure','Roomy','Shared']
sources={}; registry=[]; plotrows={}; geometry_checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
 p=W/rel;sources[rel]=sha(p);return json.loads(p.read_text())
def grid(a,axis='x'):
 a.grid(axis=axis,color='#DCE3E7',lw=.5,zorder=0);a.set_axisbelow(True)
def title(a,s):a.set_title(s,loc='left',fontweight='bold',pad=7)
def save(fig,old,rows,caption,cols,nrows=1):
 new=old+'_v22'; width,height=fig.get_size_inches();assert abs(width-(DOUBLE if cols==3 else SINGLE))<1e-6
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer(); bad=[]
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  assert t.get_fontsize()>=8.0-1e-9,(new,t.get_text(),t.get_fontsize())
  b=t.get_window_extent(renderer); fw,fh=fig.canvas.get_width_height()
  if b.x0 < -1 or b.y0 < -1 or b.x1>fw+1 or b.y1>fh+1:bad.append(t.get_text())
 assert not bad,(new,bad)
 geometry_checks.append({'figure':new,'text_outside_canvas':bad,'minimum_text_pt':min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text())})
 for ext in ['pdf','svg','png']:fig.savefig(F/(new+'.'+ext),dpi=240,facecolor='white')
 plt.close(fig);plotrows[old]=rows
 registry.append({'old_stem':old,'new_stem':new,'panel_columns':cols,'panel_rows':nrows,'span':'double' if cols==3 else 'single','width_inches':float(width),'height_inches':float(height),'caption_en':caption,'interval_definition':'As explicitly identified in caption; no invented CI','exports':[str((F/(new+'.'+ext)).relative_to(W)) for ext in ['pdf','svg','png']]})
def heat(a,vals,text,rows,cols,ttl,norm=None,cmap='Blues'):
 vals=np.asarray(vals,dtype=float);masked=np.ma.masked_invalid(vals)
 if norm is None:norm=colors.Normalize(vmin=0,vmax=1)
 cm=plt.get_cmap(cmap);a.pcolormesh(np.arange(vals.shape[1]+1),np.arange(vals.shape[0]+1),masked,norm=norm,cmap=cm,edgecolors='white',linewidth=1)
 a.set_xlim(0,vals.shape[1]);a.set_ylim(vals.shape[0],0);a.set_xticks(np.arange(len(cols))+.5,cols);a.set_yticks(np.arange(len(rows))+.5,rows);a.tick_params(length=0,pad=3)
 for i in range(vals.shape[0]):
  for k in range(vals.shape[1]):
   v=vals[i,k]
   if not np.isfinite(v):
    a.add_patch(Rectangle((k,i),1,1,facecolor='#F5F5F5',edgecolor='#A9B3BA',hatch='////',lw=.5));s='Missing'
   else:s=text[i][k]
   rgb=cm(norm(v))[:3] if np.isfinite(v) else (1,1,1);lum=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
   a.text(k+.5,i+.5,s,ha='center',va='center',fontsize=8,color='white' if lum<.53 else '#203345')
 for sp in a.spines.values():sp.set_visible(False)
 title(a,ttl)

O.mkdir(parents=True,exist_ok=True);F.mkdir(exist_ok=True)
j=read('source_data/figure_inputs/recomputed_effect_v12_20261028.json');agg={(r['case'],r['method']):r for r in j['aggregates']}
# 1. Measured service: raw processes, actual count matrix, and negative work/time contrast.
fig,aa=plt.subplots(1,3,figsize=(DOUBLE,2.90),gridspec_kw={'width_ratios':[1.6,1.08,1.0]});fig.subplots_adjust(left=.14,right=.985,top=.86,bottom=.22,wspace=.66)
a=aa[0]
for ci,c in enumerate(CASES):
 a.axhspan(ci*3-.43,ci*3+2.43,color='#EEF3F6' if ci%2==0 else 'white',zorder=0)
 for mi,m in enumerate(METHODS):
  rr=sorted([r for r in j['runs'] if r['case']==c and r['method']==m],key=lambda r:r['replicate']);assert len(rr)==5
  yy=ci*3+mi;vals=[r['measured_seconds'] for r in rr];s=agg[c,m]['metrics']['measured_seconds']
  a.scatter(vals,yy+np.linspace(-.14,.14,5),s=9,color=C[m],alpha=.55,zorder=3)
  a.errorbar(s['mean'],yy,xerr=s['stdev'],fmt=MARKS[mi],color=C[m],ms=3.4,capsize=2,lw=.9,zorder=4)
a.set_yticks(range(9),[f'{c}: {m}' for c in ['Press.','Roomy','Shared'] for m in LABELS]);a.invert_yaxis();a.set_xscale('log');a.xaxis.set_minor_formatter(NullFormatter());a.set_xticks([3,5,10,20,40],['3','5','10','20','40']);a.set_xlim(2.3,50);a.set_xlabel('Cohort time (s; log scale)');title(a,'(a) Raw runs and mean ± SD');grid(a)
vals=[[agg[c,m]['slo_completed']/agg[c,m]['completed'] for c in CASES] for m in METHODS];ct=[[f"{agg[c,m]['slo_completed']}/{agg[c,m]['completed']}" for c in CASES] for m in METHODS]
heat(aa[1],vals,ct,LABELS,['Press.','Roomy','Shared'],'(b) SLO counts')
aa[1].set_xlabel('TTFT ≤ 2 s\nmean TPOT ≤ 0.1 s',labelpad=8)
a=aa[2];reps=[agg[CASES[-1],m]['metrics']['reexecuted_tokens']['total'] for m in METHODS]
for k,m in enumerate(METHODS):
 a.hlines(k,0,reps[k],color=C[m],lw=2.2);a.scatter(reps[k],k,s=23,color=C[m],marker=MARKS[k],zorder=3);a.annotate(f'{reps[k]:,}',(reps[k],k),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
a.set_yticks(range(3),LABELS);a.set_ylim(2.5,-.6);a.set_xlim(-1000,14500);a.set_xticks([0,6000,12000],['0','6k','12k']);a.set_xlabel('Repeated positions\n(shared; 5-run totals)');title(a,'(c) Repeated work');grid(a)
save(fig,'fig05_measured_matrix',{'raw_processes':[{k:r[k] for k in ['case','method','replicate','measured_seconds','completed','slo_completed']} for r in j['runs']],'aggregation':j['aggregates'],'slo_counts':ct,'repeated_positions':dict(zip(METHODS,reps))},'Archived 45-process comparison: five fresh processes per method/scenario. (a) Every cohort time, arithmetic mean and sample SD on a logarithmic axis; shading identifies scenarios. (b) SLO-qualified/completed request counts use TTFT ≤2 s and mean TPOT ≤0.1 s. (c) Shared-case repeated executed positions summed over five processes. Zero repeats do not imply lower cohort time; methods are complete configurations, not an isolated ownership-cost intervention.',3)
# 2. Cost, work and exact output audit: vector matrices, full count denominators.
fig,aa=plt.subplots(1,3,figsize=(DOUBLE,2.78),gridspec_kw={'width_ratios':[1.1,1.35,1.1]});fig.subplots_adjust(left=.065,right=.98,top=.84,bottom=.30,wspace=.66)
vals=np.array([[agg[c,m]['metrics']['scheduler_lifecycle_seconds']['mean'] for c in CASES] for m in METHODS]);txt=[[f'{v:.3f}' for v in row] for row in vals]
heat(aa[0],vals,txt,LABELS,['Press.','Roomy','Shared'],'(a) Control time (s/run)',colors.LogNorm(vmin=.03,vmax=4),'YlGnBu');aa[0].set_xlabel('Arithmetic means; log color',labelpad=8)
a=aa[1];exe=np.array([agg[CASES[-1],m]['metrics']['executed_tokens']['total'] for m in METHODS]);rep=np.array(reps)
a.barh(range(3),exe-rep,color='#A5BCC9',height=.52,label='Nonrepeated');a.barh(range(3),rep,left=exe-rep,color=C['native_plain'],hatch='////',height=.52,label='Repeated',edgecolor='white',lw=.4)
for k,v in enumerate(exe):a.text(v+650,k,f'{v:,}',va='center',fontsize=8)
a.set_xlim(0,44500);a.set_yticks(range(3),LABELS);a.set_ylim(2.5,-.6);a.set_xticks([0,20000,40000],['0','20k','40k']);a.set_xlabel('Confirmed positions\n(shared; 5-run totals)');title(a,'(b) Executed work');grid(a);fig.legend(handles=[Rectangle((0,0),1,1,facecolor='#A5BCC9',label='Nonrepeated'),Rectangle((0,0),1,1,facecolor=C['native_plain'],hatch='////',label='Repeated')],frameon=False,fontsize=8,ncol=2,loc='lower center',bbox_to_anchor=(.51,.005),handlelength=1.2,columnspacing=1)
by={r['run_id']:r for r in j['runs']};exact=[];cnt=[]
for m in METHODS:
 v=[];tx=[]
 for c in CASES:
  if m=='native_plain':n=agg[c,m]['completed'];match=n
  else:
   pairs=[p for p in j['paired_outputs'] if p['case']==c and by[p['run_id']]['method']==m];n=sum(p['requests'] for p in pairs);match=sum(p['exact_output_matches'] for p in pairs)
  v.append(match/n);tx.append(f'{match}/{n}')
 exact.append(v);cnt.append(tx)
heat(aa[2],exact,cnt,['PRC','Native*','Hard'],['Press.','Roomy','Shared'],'(c) Exact output counts');aa[2].set_xlabel('*Native is the reference',labelpad=8)
save(fig,'fig07_cost_and_outputs',{'control_mean_seconds':vals.tolist(),'control_sample_sd_seconds':[[agg[c,m]['metrics']['scheduler_lifecycle_seconds']['stdev'] for c in CASES] for m in METHODS],'executed_positions':exe.tolist(),'repeated_positions':rep.tolist(),'exact_output_counts':cnt,'method_order':METHODS,'scenario_order':CASES},'Archived cost and output audit, retaining all unfavorable cells. (a) Arithmetic mean cumulative scheduling plus nonempty result-processing time per run (five processes); colors use a logarithmic seconds scale. This is not single-cycle latency or total CPU utilization. (b) Shared-case confirmed executed positions; hatch denotes repeated positions. (c) Exact-sequence/completed counts against matched Native; the starred Native row is a reference identity, not an independent exactness test. Different sequences remain visible even when token counts agree.',3)
# 3. True cohort pairs: no inferred repeated-run uncertainty.
h=read('source_data/figure_inputs/rechecked_holdout_20261028.json');raw=read('source_data/figure_inputs/comparison.json');pairs=[]
fig,aa=plt.subplots(1,3,figsize=(DOUBLE,2.90),gridspec_kw={'width_ratios':[1,1,1.3]});fig.subplots_adjust(left=.08,right=.975,top=.83,bottom=.28,wspace=.58)
for ci,case in enumerate(['HOLDOUT_dolly_pressure','HOLDOUT_dolly_roomy']):
 a=aa[ci]
 for k in range(3):
  r0=next(r for r in raw['runs'] if r['case']==case and r['run_id'].endswith(f'prc_original_c{k}'));r1=next(r for r in raw['runs'] if r['case']==case and r['run_id'].endswith(f'prc_tuned_c{k}'));x0=r0['measured_seconds'];x1=r1['measured_seconds']
  a.axhspan(k-.38,k+.38,color='#EEF3F6',zorder=0);a.plot([x1,x0],[k,k],color='#9FB3C1',lw=2);a.scatter(x0,k,s=24,facecolor='white',edgecolor=C['prc_original'],zorder=3);a.scatter(x1,k,s=24,color=C['prc_tuned'],zorder=3)
  a.text((x0+x1)/2,k-.23,f'{r0["slo_completed"]}→{r1["slo_completed"]}/16',ha='center',fontsize=8)
  a.text((x0+x1)/2,k+.28,f'Δt {100*(x1/x0-1):+.1f}%',ha='center',fontsize=8,color='#455F70')
  pairs.append({'condition':case,'cohort':k+1,'D64_seconds':x0,'D256_seconds':x1,'SLO64':r0['slo_completed'],'SLO256':r1['slo_completed'],'delta_time_percent':100*(x1/x0-1)})
 a.set_yticks(range(3),['C1','C2','C3']);a.set_ylim(2.58,-.58);a.set_xlim(1.8,6.8);a.set_xticks([2,4,6]);a.set_xlabel('Cohort time (s)');title(a,['(a) Pressure: paired inputs','(b) Roomy: paired inputs'][ci]);grid(a)
aa[0].legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=C['prc_original'],label='$D=64$'),Line2D([],[],marker='o',ls='',c=C['prc_tuned'],label='$D=256$')],frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(-.04,-.37),columnspacing=.65,handlelength=.8)
a=aa[2];vs=['prc_original','prc_tuned','native_plain','hard_reserve'];vl=['PRC $D=64$','PRC $D=256$','Native','Hard']
for ci,case in enumerate(['HOLDOUT_dolly_pressure','HOLDOUT_dolly_roomy']):
 for k,v in enumerate(vs):
  g=next(g for g in h['aggregates'] if g['case']==case and g['variant']==v);y=k+(ci-.5)*.25;a.scatter(g['requests_per_second'],y,s=23,marker=['o','s'][ci],color=C[v],zorder=3);a.text(g['requests_per_second']+.13,y,f'{g["requests_per_second"]:.2f}',va='center',fontsize=8)
a.set_yticks(range(4),vl);a.invert_yaxis();a.set_xlim(2.4,8.3);a.set_xticks([3,5,7]);a.set_xlabel('Pooled completions / s');title(a,'(c) All configurations');grid(a);a.legend(handles=[Line2D([],[],marker=m,ls='',c='#455F70',label=l) for m,l in [('o','Pressure'),('s','Roomy')]],frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(-.04,-.37),columnspacing=.65,handlelength=.8)
save(fig,'fig06_holdout',{'cohort_pairs':pairs,'aggregates':h['aggregates']},'Frozen drain-trigger comparison on 48 new identities reused across 24 processes. (a,b) Lines join each distinct input cohort under D=64 (open) and D=256 (filled); upper labels give SLO-qualified/16 counts and lower labels the cohort-time percentage change. Cohort dispersion is not repeated-run uncertainty. (c) Pooled completed requests divided by summed process time, with pressure circles and roomy squares. Four of six direct D64/D256 pairs have output differences; these are complete-configuration sensitivities, not a scheduling-only causal effect or cross-GPU balancing result.',3)
# 4. Client-clock stages and actual receipt/completion staircases, two stacked single-column panels.
tl=read('data/derived_client_trajectory.json')
fig,aa=plt.subplots(2,1,figsize=(SINGLE,4.22),gridspec_kw={'height_ratios':[1.35,1]});fig.subplots_adjust(left=.19,right=.87,top=.90,bottom=.12,hspace=.66)
for i,m in enumerate(tl['timeline']):
 col=[C['prc'],C['green']][int(m['worker'][-1])];a=aa[0]
 a.plot([m['offered_s'],m['admitted_s']],[i,i],color='#B3BBC0',lw=3);a.plot([m['admitted_s'],m['completed_s']],[i,i],color=col,lw=3)
 a.scatter(m['admitted_s'],i,s=13,fc='white',ec=col);a.scatter(m['first_client_receipt_s'],i,s=24,marker='|',c='black');a.scatter(m['completed_s'],i,s=14,marker='s',c=col);a.annotate(str(m['delivered_tokens']),(m['completed_s'],i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
a=aa[0];a.set_yticks(range(8),[f"R{i} W{m['worker'][-1]}" for i,m in enumerate(tl['timeline'])]);a.invert_yaxis();a.set_xlabel('Client time from trace start (s)');a.set_xlim(0,max(x['completed_s']for x in tl['timeline'])*1.13);a.set_xticks([0,5,10]);title(a,'(a) Observed request stages');grid(a)
a=aa[1];r=tl['cumulative_receipts'];a.step(r['time_s'],r['total_tokens'],where='post',c='#253B4A',label='Total')
for k in range(2):a.step(r['time_s'],r['worker_tokens'][k],where='post',c=[C['prc'],C['green']][k],ls=['--',':'][k],label=f'W{k}')
b=a.twinx();b.step([0]+tl['completion_time_s'],[0]+tl['completion_counts'],where='post',c=C['gold'],ls='-.');b.set_ylabel('Completed requests',color=C['gold'],fontsize=8);b.set_ylim(0,8.7);b.set_yticks([0,4,8]);a.set_xlabel('Client time from trace start (s)');a.set_ylabel('Delivered tokens');a.set_ylim(0,190);a.set_xticks([0,5,10]);a.legend(frameon=False,ncol=3,loc='upper left',fontsize=8);title(a,'(b) Receipt / completion steps');grid(a,'y')
save(fig,'fig08_two_worker_execution',tl,'Client-observed two-engine trajectory on one RTX5090. All times and cumulative counts come from the exact derived trajectory, not invented raw events. Gray denotes offered-to-admitted waiting; worker colors denote admitted-to-final-delivery intervals. The gold right axis counts final deliveries. This is not internal token-kernel timing or heterogeneous-GPU speedup.',1,2)
# 5. Snapshot diagnostic: condition ranges are not independent-run uncertainty.
j=read('source_data/fig04_ownership.json');rows=j['diagnostic_rows'];ms=['native_shadow','shared_closure','shared_incremental'];labs=['Shadow','Full closure','Incremental'];ks=['E7_ownership_forced','E7_ownership_eos'];pr=[]
fig=plt.figure(figsize=(SINGLE,3.10));a=fig.add_axes([.25,.59,.715,.29]);tab=fig.add_axes([.25,.08,.715,.29])
for i,m in enumerate(ms):
 rr=[next(r for r in rows if r['method']==m and r['case']==k) for k in ks];vals=[100*r['gain_snapshots']/r['snapshots'] for r in rr];a.hlines(i,min(vals),max(vals),color='#AABBC6',lw=2,zorder=2)
 for k,r in enumerate(rr):
  rate=vals[k];a.scatter(rate,i,s=38 if k==0 else 18,facecolor='white' if k==0 else C['gold'],edgecolor=C['prc'] if k==0 else C['gold'],linewidth=1.1,marker=['o','s'][k],zorder=3+k);pr.append({**r,'gain_percent':rate})
a.set_yticks(range(3),labs);a.set_ylim(2.5,-.5);a.set_xlim(.25,.44);a.set_xticks([.25,.30,.35,.40],['.25','.30','.35','.40']);a.set_xlabel('Gain / recorded snapshots (%)');grid(a);title(a,'(a) Observed condition range')
a.legend(handles=[Line2D([],[],marker=mk,ls='',c=col,mfc='white' if mk=='o' else col,label=lab) for mk,col,lab in [('o',C['prc'],'Forced cap'),('s',C['gold'],'EOS enabled')]],frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.31,1.13),columnspacing=.7,handlelength=.8)
tab.axis('off');data=[]
for m in ms:
 rr=[next(r for r in rows if r['method']==m and r['case']==k) for k in ks];data.append([f"{r['gain_snapshots']}/{r['snapshots']:,}" for r in rr])
t=tab.table(cellText=data,rowLabels=labs,colLabels=['Forced cap','EOS enabled'],cellLoc='center',rowLoc='right',loc='center',colWidths=[.5,.5]);t.auto_set_font_size(False);t.set_fontsize(8);t.scale(1,1.12)
for (r,c),cell in t.get_celld().items():cell.set_edgecolor('#DCE3E7');cell.set_linewidth(.4);cell.set_facecolor('#EEF3F6' if r==0 else 'white')
tab.set_title('(b) Exact gain / snapshot counts',loc='left',fontweight='bold',fontsize=9,pad=4)
save(fig,'fig05_C_single_column',{'diagnostic_rows':pr,'counts_table':data,'correlated_gain_observations':58,'distinct_positive_rank_labeled_states':11,'independent_runs':6,'uncertainty':'Horizontal segments span the two observed settings; not error bars or CI. Snapshots within a run are correlated.'},'Constructed ownership diagnostic from six 12-request runs. (a) Points give gain/recorded-snapshot percentages; connecting segments span the two measured settings and are not uncertainty intervals. (b) Exact numerators and denominators preserve unequal snapshot counts. The 58 correlated gain observations represent 11 distinct positive rank-labeled states; no independent-snapshot inference is made. All 72 requests, including the 36 EOS-enabled requests, stop at their caps.',1,2)

# Two supporting panels are maintained in a small adjacent module.
from redesign_legacy_hardware_theory_v22 import generate
generate(W)
sub=D/'v22_figures/legacy_hardware_theory'
for r in json.loads((sub/'layoutcaptions.json').read_text()):
 r['caption_en']=r['caption'];registry.append(r)
for child in sub.glob('*_plotrows.json'):plotrows[child.name.removesuffix('_plotrows.json')]=json.loads(child.read_text())
childsources=json.loads((sub/'sourcehash.json').read_text())
for r in childsources['inputs']:sources[r['path']]=r['sha256']
for r in registry:
 r.update(rows=r['panel_rows'],cols=r['panel_columns'],width_in=r['width_inches'],height_in=r['height_inches'],caption_plain=r['caption_en'],caption_tex=r['caption_en'].replace('≤',r'$\le$'),plot_data_path='source_data/v22_figures/legacy/plotrows.json',source_sha256=dict(sources),figure_sha256=sha(F/(r['new_stem']+'.pdf')),minimum_font_pt=8.0)
 r['asset_sha256']={ext:sha(F/(r['new_stem']+'.'+ext)) for ext in ['pdf','svg','png']}
(O/'plotrows.json').write_text(json.dumps(plotrows,indent=2))
(O/'sourcehash.json').write_text(json.dumps({'raw_sources_sha256':sources,'redraw_scripts_sha256':{str(Path(__file__).relative_to(W)):sha(Path(__file__)),'figure_source/redesign_legacy_hardware_theory_v22.py':sha(W/'figure_source/redesign_legacy_hardware_theory_v22.py')},'old_sources_and_data_immutable':True},indent=2))
(O/'layoutcaptions.json').write_text(json.dumps(registry,indent=2))
(O/'registry.json').write_text(json.dumps({'figures':registry,'plot_data_path':'source_data/v22_figures/legacy/plotrows.json','sources_path':'source_data/v22_figures/legacy/sourcehash.json','qa_path':'source_data/v22_figures/legacy/PDF_EXPORT_QA.json'},indent=2))
(O/'geometry_checks.json').write_text(json.dumps(geometry_checks,indent=2))
print(json.dumps({'figures':len(registry),'registry':str(O/'layoutcaptions.json'),'geometry':geometry_checks},indent=2))
