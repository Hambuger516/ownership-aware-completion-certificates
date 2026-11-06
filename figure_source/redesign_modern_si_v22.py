"""Ten modern SI figures, fixed final dimensions, all existing measurements.

No GPU actions, serving changes, smoothing, invented intervals or density plots.
Every render has an English caption, exact plotted values and source SHA256.
"""
from pathlib import Path
import collections,csv,datetime,hashlib,json,math,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
from matplotlib.ticker import NullFormatter,MaxNLocator

W=Path(__file__).resolve().parents[1];O=W/'figs';D=W/'source_data/v22_figures/modern';D.mkdir(parents=True,exist_ok=True)
SW=3.16697799916978;DW=6.500000276740003
for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'lines.linewidth':.9})
C={'inc':'#687E88','fixed':'#A37232','adaptive':'#285F8C','native':'#28776F','banker_c0':'#A37232','hard_reserve':'#687E88','reuse':'#285F8C'}
MARK={'inc':'o','fixed':'^','adaptive':'D','native':'s','reuse':'D'}
NAME={'inc':'INC','fixed':'Fixed INC-Q','adaptive':'Adaptive INC-Q','native':'Native'}
ALLSRC={};REG=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(n):
 p=W/n;ALLSRC[n]=sha(p)
 if p.suffix=='.csv':
  with p.open()as f:return list(csv.DictReader(f))
 return json.loads(p.read_text())
def gm(v):return math.exp(statistics.mean(math.log(float(x))for x in v))
def truth(v):return v is True or v=='True'
def clean(a,axis='both'):
 a.grid(axis=axis,color='#DCE3E8',alpha=.65,lw=.4);a.set_axisbelow(True);a.tick_params(length=2.5,width=.6)
def parity(a,axis='x'):
 (a.axvline if axis=='x'else a.axhline)(1,color='#45535D',ls='--',lw=.65,zorder=0)
def logticks(a,axis,values):
 ax=a.xaxis if axis=='x'else a.yaxis;lo,hi=a.get_xlim()if axis=='x'else a.get_ylim();ticks=[x for x in values if min(lo,hi)<=x<=max(lo,hi)]
 (a.set_xticks if axis=='x'else a.set_yticks)(ticks,labels=[f'{x:g}'for x in ticks]);ax.set_minor_formatter(NullFormatter())
def strip(a,values,y,col,mark='D',offset=.13):
 v=list(map(float,values));n=len(v);q1,q3=np.quantile(v,[.25,.75]);a.plot([min(v),max(v)],[y,y],c=col,lw=.85,zorder=2);a.plot([q1,q3],[y,y],c=col,lw=2.4,alpha=.55,zorder=2)
 a.scatter(v,[y+(j-(n-1)/2)*(2*offset/max(n-1,1))for j in range(n)],s=12,fc='white',ec=col,lw=.55,zorder=3);a.plot(gm(v),y,mark,c=col,ms=3.2,zorder=4)
 return dict(values=v,n=n,GM=gm(v),minimum=min(v),maximum=max(v),Q1=float(q1),Q3=float(q3))
def save(fig,old,rows,cols,caption,data,sources):
 span='double'if cols==3 else'single';expected=DW if span=='double'else SW;assert math.isclose(fig.get_size_inches()[0],expected,abs_tol=1e-8)
 stem=old+'_v22';files={}
 for ext in ['pdf','svg','png']:
  p=O/(stem+'.'+ext);fig.savefig(p,dpi=260,bbox_inches=None);files[ext]=dict(path='figs/'+p.name,sha256=sha(p))
 texts=[t.get_fontsize()for ax in fig.axes for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]if t.get_text()]
 assert min(texts)>=8.0,(old,min(texts))
 width,height=map(float,fig.get_size_inches());plt.close(fig)
 source_hash={n:ALLSRC.get(n)or sha(W/n)for n in sources}
 for n,h in source_hash.items():ALLSRC[n]=h
 dp=D/(old+'_plotrows.json');dp.write_text(json.dumps(dict(figure=old,caption=caption,statistics='Raw paired process points; geometric means, empirical quartiles and observed ranges only unless caption states another exact counting convention.',source_sha256=source_hash,plot_data=data),indent=2)+'\n')
 REG.append(dict(old_stem=old,new_stem=stem,rows=rows,cols=cols,span=span,width_in=width,height_in=height,minimum_font_pt=min(texts),caption=caption,caption_plain=caption,caption_tex=caption,plot_data=str(dp.relative_to(W)),plot_data_path=str(dp.relative_to(W)),plot_data_sha256=sha(dp),source_sha256=source_hash,outputs=files,figure_sha256={k:v['sha256']for k,v in files.items()},recipe_sha256=sha(Path(__file__)),fixed_geometry=True,vector_PDF=True,live_text_SVG=True,native_PPT_claim=False))
def case_label(c):
 return f"P{int(c.split('_')[0][-2:])+1} "+{'front':'F','interleave':'I','back':'B'}[c.rsplit('_',1)[-1]]


def old_replication():
 paths=['source_data/a800_verified/'+n for n in ['all36_primary_pairs.csv','all18_Native_rows.csv','all9_mechanism_pairs.csv','all108_service_rows.csv','FINAL_NUMERIC_CHECK.json']]
 primary,native,mechanism,services,check=[get(p)for p in paths];assert check['status']=='INDEPENDENT_COMPLETE132_ATTEMPTS129_QUALIFIED_NUMERIC_AND_REGISTERED_BYTES_CHECK_PASS'
 cases=sorted({r['case']for r in primary if r['family']=='main'},key=lambda c:(c.split('_')[0],['front','interleave','back'].index(c.rsplit('_',1)[-1])))
 fig,axs=plt.subplots(2,1,figsize=(SW,5.3));fig.subplots_adjust(left=.21,right=.87,bottom=.09,top=.95,hspace=.50);data=[]
 for a,metric,title in zip(axs,['M_ratio','brief_ratio'],['(a) Cohort completion','(b) Brief completion']):
  vv=[float(r[metric])for r in primary if r['family']=='main'];lo=min(.96,min(vv)*.88);hi=max(1.05,max(vv)*1.10)
  for y,c in enumerate(cases):
   if y//3%2==0:a.axhspan(y-.48,y+.48,color='#EDF1F3',zorder=-2)
   r=sorted([z for z in primary if z['family']=='main'and z['case']==c],key=lambda z:int(z['repeat']));v=[float(z[metric])for z in r];s=strip(a,v,y,C['adaptive']);a.text(1.02,y,f"{s['GM']:.3f}",transform=a.get_yaxis_transform(),ha='left',va='center',fontsize=8)
   data.append(dict(case=c,metric=metric,source_rows=r,**s))
  a.set(xscale='log',xlim=(lo,hi),ylim=(8.65,-.65),yticks=range(9),yticklabels=[case_label(c)for c in cases],xlabel='INC / production C0',title=title);a.text(1.02,1.015,'GM',transform=a.transAxes,fontsize=8);parity(a);clean(a,'x');logticks(a,'x',[.5,.7,.9,1,1.3]if metric=='M_ratio'else[.05,.1,.3,1])
 caption='Earlier A800 confirmation, with all nine shared-prefix scenarios and three process pairs per scenario. Rows retain front/interleaved/back arrivals for each document pair. Open points are paired process ratios; diamonds are geometric means, thick segments empirical interquartile ranges and thin segments observed minima/maxima. Right-hand values give geometric means. Shading identifies document-pair groups. The dashed line denotes equal time to production C0; no confidence interval or workload-population inference is shown.'
 save(fig,'fig06_a800_confirmation',2,1,caption,data,paths)
 # Joint work/time scatter and a vector exact-output matrix.
 fig,axs=plt.subplots(2,1,figsize=(SW,4.5),gridspec_kw={'height_ratios':[2,1]});fig.subplots_adjust(left=.20,right=.96,bottom=.12,top=.94,hspace=.60);a,b=axs;points=[]
 for family,mk in [('main','o'),('alpaca','s')]:
  for r in [z for z in native if z['family']==family]:
   by={arm:next(x for x in services if x['family']==family and x['case']==r['case']and x['repeat']==r['repeat']and x['arm']==arm)for arm in ['inc_closed','native_plain']};x=float(by['inc_closed']['real_positions'])/float(by['native_plain']['real_positions']);y=float(r['INC_over_Native_M_ratio']);col=C['adaptive']if family=='main'else C['fixed'];a.scatter(x,y,s=21,marker=mk,fc=col if truth(r['all16_outputs_equal'])else'white',ec=col,lw=.6);points.append(dict(source=r,work_ratio=x,time_ratio=y,paired_service_rows=by))
 a.axvspan(0,1,color=C['adaptive'],alpha=.045);a.axhline(1,c='#45535D',ls='--',lw=.65);a.axvline(1,c='#45535D',ls='--',lw=.65);a.set(xlim=(min(p['work_ratio']for p in points)*.9,1.05),ylim=(.85,max(p['time_ratio']for p in points)*1.12),xlabel='Executed positions / Native',ylabel='Cohort time / Native',title='(a) Work saved and service cost');clean(a)
 a.legend(handles=[Line2D([],[],marker='o',ls='',c=C['adaptive'],label='Shared prefix'),Line2D([],[],marker='s',ls='',c=C['fixed'],label='Instructions')],frameon=False,loc='upper right',fontsize=8)
 mainr=sorted([r for r in native if r['family']=='main'],key=lambda r:cases.index(r['case']));alpr=sorted([r for r in native if r['family']=='alpaca'],key=lambda r:(r['case'],int(r['repeat'])));z=np.array([[int(r['matched_requests'])for r in mainr],[int(r['matched_requests'])for r in alpr]])
 b.pcolormesh(np.arange(10),np.arange(3),z,cmap='Blues',vmin=0,vmax=16,edgecolors='white',linewidth=.6,rasterized=False)
 for i in range(2):
  for j in range(9):b.text(j+.5,i+.5,str(z[i,j]),ha='center',va='center',color='white'if z[i,j]>=10 else'#183747',fontsize=8)
 b.set(xticks=np.arange(9)+.5,xticklabels=range(1,10),yticks=[.5,1.5],yticklabels=['Prefix','Instr.'],ylim=(2,0),xlabel='Ordered comparison block',title='(b) Exact token matches / 16');b.tick_params(length=0)
 caption='Earlier A800 Native comparisons. Panel (a) retains all 18 paired observations: circles are shared-prefix inputs and squares instruction inputs; filled markers denote exact agreement for all 16 output sequences. The pale region marks fewer executed model positions than Native; the dashed axes mark equal work and equal cohort time. Saved positions are not FLOPs or GPU-time savings. Panel (b) is a vector matrix of exact matches out of 16, ordered by document pair/arrival for prefix inputs and by cohort/repetition for instructions. Every comparison and adverse time ratio is retained.'
 save(fig,'fig07_a800_native',2,1,caption,dict(points=points,matrix=z.tolist(),matrix_row_records=[mainr,alpr]),paths)
 # Full-information mechanism plus the independent-source workload.
 fig,axs=plt.subplots(2,1,figsize=(SW,5.3),gridspec_kw={'height_ratios':[1.8,1]});fig.subplots_adjust(left=.23,right=.88,bottom=.17,top=.94,hspace=.56);a,b=axs;matrix=[];mdata=[]
 for c in cases:
  pair={arm:next(r for r in services if r['case']==c and r['stage']=='mechanism'and r['arm']==arm)for arm in ['full_c1','full_c0']};v=[float(pair['full_c1'][m])/float(pair['full_c0'][m])for m in ['M_s','brief_mean_s']];stored=next(r for r in mechanism if r['case']==c);assert math.isclose(v[0],float(stored['fullC1_over_fullC0_M_ratio']),rel_tol=1e-12);matrix.append(v);mdata.append(dict(case=c,ratios=v,source_rows=pair))
 z=np.log2(matrix);lim=max(abs(z.min()),abs(z.max()),.05);cmap=colors.LinearSegmentedColormap.from_list('ratio',['#285F8C','#FFFFFF','#A37232']);im=a.pcolormesh(np.arange(3),np.arange(10),z,cmap=cmap,norm=colors.TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),edgecolors='#DCE3E8',linewidth=.4,rasterized=False);cb=fig.colorbar(im,ax=a,fraction=.04,pad=.04);cb.set_ticks([-lim,0,lim],labels=[f'{2**(-lim):.2g}','1',f'{2**lim:.2g}']);cb.solids.set_rasterized(False);cb.ax.tick_params(labelsize=8,length=2)
 for i,row in enumerate(matrix):
  for j,v in enumerate(row):a.text(j+.5,i+.5,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white'if abs(math.log2(v))>.65*lim else'#243747')
 a.set(xticks=[.5,1.5],xticklabels=['Cohort','Brief'],yticks=np.arange(9)+.5,yticklabels=[case_label(c)for c in cases],ylim=(9,0),title='(a) Full C1 / full C0');a.tick_params(length=0)
 alpcases=sorted({r['case']for r in primary if r['family']=='alpaca'});adata=[]
 for y,c in enumerate(alpcases):
  for key,col,off,mark in [('M_ratio',C['adaptive'],-.18,'D'),('brief_ratio',C['native'],.18,'s')]:
   rr=sorted([r for r in primary if r['case']==c],key=lambda r:int(r['repeat']));s=strip(b,[r[key]for r in rr],y+off,col,mark,offset=.06);adata.append(dict(case=c,metric=key,source_rows=rr,**s))
 b.set(yscale='linear',xscale='log',ylim=(2.55,-.55),yticks=range(3),yticklabels=['A1','A2','A3'],xlabel='INC / production C0',title='(b) Independent-source input');parity(b);clean(b,'x');fig.legend(handles=[Line2D([],[],marker='D',c=C['adaptive'],ls='',label='Cohort'),Line2D([],[],marker='s',c=C['native'],ls='',label='Brief')],frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.56,.015));b.xaxis.set_minor_formatter(NullFormatter());logticks(b,'x',[.8,.85,.9,.95,1,1.05,1.1])
 caption='Earlier matched-information-cost control and instruction-family comparison. Panel (a) reports each single matched full-C1/full-C0 service pair for all nine document/arrival cells, for both cohort and brief completion. Cell colors use one symmetric log2-ratio scale centered at equality at 1 (blue below, ochre above); annotations retain raw ratios. Panel (b) retains three process pairs in each instruction cohort: open points, geometric-mean markers, empirical interquartile segments and observed ranges. Both full views pay the same checking contract; single-pair mechanism cells are not repeated-run distributions.'
 save(fig,'fig08_a800_mechanism_alpaca',2,1,caption,dict(mechanism=mdata,independent=adata,ratio_color_center=1,log2_color_halfspan=lim),paths)


def diagnostics():
 paths=['source_data/v15_verified_updates/DIAGNOSTIC_READOUT_INTERPRETED.json','experiments/new_verified_evidence/study_scripts/W37/all_primary_pairs.csv','experiments/new_verified_evidence/study_scripts/W40/all_primary_pairs.csv','source_data/a800_verified/all108_service_rows.csv']
 diag,p37,p40,services=[get(p)for p in paths];fig,axs=plt.subplots(1,3,figsize=(DW,2.65));fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.88,wspace=.58);a,b,c=axs;dist=[]
 for ci,case in enumerate(diag['cases']):
  for run in case['runs']:
   counts={int(k):v for k,v in run['batch_request_count_distribution'].items()};assert sum(counts.values())==run['positive_model_batches'];xx=sorted(counts);yy=np.cumsum([counts[x]for x in xx])/sum(counts.values());a.step([0,*xx],[0,*yy],where='post',c=C['inc']if run['arm']=='inc'else C['native'],ls='-'if ci==0 else'--',label=('P'if ci==0 else'I')+' '+('INC'if run['arm']=='inc'else'Native'));dist.append(dict(case=case['case'],arm=run['arm'],counts=counts,cumulative=yy.tolist(),batch_counts=sum(counts.values())))
 a.set(xlim=(0,16),ylim=(0,1.02),xticks=[1,4,8,16],xlabel='Requests / model batch',ylabel='Cumulative batch fraction',title='(a) Batch composition');a.legend(frameon=False,fontsize=8,loc='lower right');clean(a)
 cost=[]
 for y,(name,rr)in enumerate([('INC / full',p37),('Cache on / off',p40)]):
  assert len(rr)==6;s=strip(b,[r['M_ratio']for r in rr],y,C['adaptive'],offset=.10);cost.append(dict(control=name,source_rows=rr,**s))
 b.set(yticks=[0,1],yticklabels=['INC/full','Cache/off'],ylim=(1.55,-.55),xlabel='Paired cohort-time ratio',title='(b) Check-cost controls');parity(b);clean(b,'x');b.set_xticks([.970,.985,1],labels=['0.970','0.985','1'])
 owner=[]
 for y in range(3):
  case=f'confirm{y:02}_interleave';rr={arm:next(r for r in services if r['case']==case and r['stage']=='mechanism'and r['arm']==arm)for arm in ['full_c1','full_c0']};v=float(rr['full_c1']['brief_mean_s'])/float(rr['full_c0']['brief_mean_s']);c.plot([v,1],[y,y],c='#D1D9DE',lw=1.4);c.plot(v,y,'D',c=C['fixed'],ms=4);c.text(v,y-.18,f'{v:.3f}',ha='center',fontsize=8);owner.append(dict(case=case,ratio=v,source_rows=rr))
 c.set(xscale='log',xlim=(.04,1.15),ylim=(2.6,-.6),yticks=range(3),yticklabels=['P1 I','P2 I','P3 I'],xlabel='C1 / C0 brief time',title='(c) Ownership view');parity(c);clean(c,'x');logticks(c,'x',[.05,.1,.3,1])
 caption='Historical diagnosis from separately scoped studies. Panel (a) is an exact empirical cumulative batch-size distribution from the four profiled diagnostic services; P/I denote prefix/instruction inputs. It is not an efficacy timing comparison. Panel (b) shows all six matched process ratios for each full-versus-incremental and refusal-cache control: three pairs in each of two fixed inputs. Geometric means weight the inputs equally; thick segments are empirical interquartile ranges and thin segments observed ranges. Panel (c) retains the single earlier full-C1/full-C0 brief-completion ratio for each interleaved document pair. The three panels are distinct populations and are not pooled or interpreted as a causal CPU/GPU decomposition.'
 save(fig,'fig09_runtime_diagnostics',1,3,caption,dict(batch_distributions=dist,cost_controls=cost,ownership=owner),paths)


def sensitivity():
 path='source_data/v16_verified/w52_services.csv';rs=[r for r in get(path)if r['phase']=='timing'];k={(r['case'],int(r['repeat']),r['arm']):r for r in rs};cases=['confirm02_back','alpaca00_interleave'];levels=[1,2,3,4,6,8];fig,axs=plt.subplots(2,3,figsize=(DW,4.65));fig.subplots_adjust(left=.08,right=.985,bottom=.11,top=.88,hspace=.56,wspace=.55);data=[]
 for row,case in enumerate(cases):
  for col,metric in enumerate(['M_s','brief_s','requests_per_model_batch']):
   a=axs[row,col];isratio=col<2
   for q,colr,mk in [('off',C['inc'],'s'),('on',C['adaptive'],'o')]:
    means=[];low=[];high=[]
    for level in levels:
     rr=[k[(case,j,f'k{level}_{q}')]for j in range(3)];v=[float(r[metric])/(float(k[(case,j,'native')][metric])if isratio else 1)for j,r in enumerate(rr)];means.append(gm(v)if isratio else statistics.mean(v));low.append(min(v));high.append(max(v));a.scatter([level]*3,v,s=8,fc='white',ec=colr,lw=.5,zorder=4);data.append(dict(case=case,kappa=level,queue_scan=q,metric=metric,values=v,center=means[-1],minimum=min(v),maximum=max(v),source_rows=rr))
    a.fill_between(levels,low,high,color=colr,alpha=.12);a.plot(levels,means,c=colr,marker=mk,ms=3,lw=.9)
   base=[float(k[(case,j,'base')][metric])/(float(k[(case,j,'native')][metric])if isratio else 1)for j in range(3)];bc=gm(base)if isratio else statistics.mean(base);a.axhspan(min(base),max(base),color=C['fixed'],alpha=.08);a.axhline(bc,c=C['fixed'],ls=':',lw=.9)
   data.append(dict(case=case,kappa=None,queue_scan='gate_off_reference',metric=metric,values=base,center=bc))
   if isratio:
    a.set_yscale('log');parity(a,'y');a.yaxis.set_minor_formatter(NullFormatter());logticks(a,'y',[1,2,4,8]if col==0 else[.03,.1,.3,1,3,10,30,100])
   a.set(xlim=(.65,8.4),xticks=levels,xlabel=r'$\kappa_{\rm adm}$',ylabel=['Cohort / Native','Brief / Native','Requests / model batch'][col],title=f"({'abcdef'[row*3+col]}) "+('Prefix'if row==0 else'Instructions'));clean(a)
 fig.legend(handles=[Line2D([],[],c=C['inc'],marker='s',label='Scan off',ms=3),Line2D([],[],c=C['adaptive'],marker='o',label='Scan on',ms=3),Line2D([],[],c=C['fixed'],ls=':',label='Gate-off reference')],loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False)
 caption='Historical six-level admission sensitivity on the two exposed inputs. Rows separate prefix and instruction cohorts; columns report cohort time relative to same-block Native, brief completion relative to Native, and actual mean requests per positive model batch. All three process repetitions at each measured gate values 1/2/3/4/6/8 are shown. Ratio centers are geometric means and batch-width centers arithmetic means. Curves connect measured centers only; shading is the observed minimum-to-maximum process range, not a confidence band. Ochre references are the measured gate-off deployment, with its observed range. Dashed equality at 1 applies only to the two ratio columns. No unmeasured parameter optimum or workload-population inference is implied.'
 save(fig,'fig_sensitivity_v16',2,3,caption,data,[path,'source_data/v16_verified/w52_final.json'])


def development():
 path='source_data/v17_verified/development_all_services.csv';rs=[r for r in get(path)if r['phase']=='timing'];k={(r['case'],int(r['quota_factor']),int(r['repeat']),r['arm']):r for r in rs};cases=['confirm02_back','alpaca00_interleave'];arms=['inc','fixed','adaptive','native'];data=[]
 fig,axs=plt.subplots(2,3,figsize=(DW,4.4),sharex=True,sharey=True);fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.86,hspace=.58,wspace=.30)
 for row,case in enumerate(cases):
  for col,q in enumerate([1,2,4]):
   a=axs[row,col]
   for arm in arms:
    rr=[k[(case,q,j,arm)]for j in range(4)];xx=[float(r['M_s'])/float(k[(case,q,j,'native')]['M_s'])for j,r in enumerate(rr)];yy=[float(r['brief_s'])/float(k[(case,q,j,'native')]['brief_s'])for j,r in enumerate(rr)];a.add_patch(Rectangle((min(xx),min(yy)),max(xx)-min(xx),max(yy)-min(yy),fc=C[arm],ec=C[arm],alpha=.13,lw=.7));a.scatter(xx,yy,s=11,fc='white',ec=C[arm],marker=MARK[arm],lw=.55,zorder=3);a.plot(gm(xx),gm(yy),MARK[arm],c=C[arm],ms=4,zorder=4);data.append(dict(case=case,quota=q,arm=arm,M_values=xx,brief_values=yy,GM_M=gm(xx),GM_brief=gm(yy),source_rows=rr))
   a.set(xscale='log',yscale='log',xlim=(.85,4.6),ylim=(.025,6.0),xlabel='Cohort time / Native',title=f"({'abcdef'[row*3+col]}) "+('Prefix'if row==0 else'Instr.')+f', {q}'+r'$\times$');a.axvspan(.85,1,color=C['native'],alpha=.04);parity(a);parity(a,'y');clean(a);logticks(a,'x',[1,2,4]);logticks(a,'y',[.03,.1,.3,1,3])
   if col==0:a.set_ylabel('Brief completion / Native')
 fig.legend(handles=[Line2D([],[],c=C[x],marker=MARK[x],ls='',ms=3.5,label=NAME[x])for x in arms],ncol=4,loc='upper center',bbox_to_anchor=(.5,.99),frameon=False)
 caption='Earlier adaptive-development operating region, with the three measured quotas as columns and the two exposed inputs as rows. Each marker is one of four same-card process comparisons to Native; larger markers are coordinatewise geometric means. Translucent rectangles bound the observed paired cohort/brief outcomes, not confidence regions or a Pareto frontier. Equality lines identify the Native reference; pale left-hand regions indicate lower cohort time. All methods, raw pairs and adverse outcomes are retained. This development timing population is separate from frozen new-input confirmation and projection-reuse studies.'
 save(fig,'fig_quota_v17',2,3,caption,data,[path])
 # Positive model batches as vector matrix plus the actual wall-interval ledger.
 fig,axs=plt.subplots(2,1,figsize=(SW,4.6),gridspec_kw={'height_ratios':[1.15,1]});fig.subplots_adjust(left=.27,right=.92,bottom=.17,top=.93,hspace=.66);a,b=axs;cells=[(c,q)for c in cases for q in [1,2,4]];counts=[[statistics.mean(int(k[(c,q,j,arm)]['model_batches'])for j in range(4))for c,q in cells]for arm in arms];z=np.array(counts);mesh=a.pcolormesh(np.arange(7),np.arange(5),z,cmap='Blues',norm=colors.LogNorm(vmin=z.min(),vmax=z.max()),edgecolors='white',linewidth=.7,rasterized=False);cb=fig.colorbar(mesh,ax=a,fraction=.035,pad=.03);cb.set_ticks([250,1000,2000],labels=['250','1k','2k']);cb.solids.set_rasterized(False);cb.ax.tick_params(labelsize=8,length=2);cb.minorticks_off()
 for y,row in enumerate(counts):
  for x,v in enumerate(row):a.text(x+.5,y+.5,f'{v:g}',ha='center',va='center',fontsize=8,color='white'if v>850 else'#183747')
 a.set(xticks=np.arange(6)+.5,xticklabels=['P1','P2','P4','I1','I2','I4'],yticks=np.arange(4)+.5,yticklabels=['INC','Fixed','Adaptive','Native'],ylim=(4,0),xlabel='Input / quota multiplier',title='(a) Mean positive model batches');a.tick_params(length=0)
 ledger=[]
 for y,(case,arm)in enumerate([(cases[0],'native'),(cases[0],'adaptive'),(cases[1],'native'),(cases[1],'adaptive')]):
  rr=[k[(case,4,j,arm)]for j in range(4)];vv=[float(r['M_s'])for r in rr];total=statistics.mean(vv);s=statistics.mean(float(r['schedule_wall_s'])for r in rr);b.barh(y,s,color=C['adaptive'],height=.52);b.barh(y,total-s,left=s,color='#D9E0E5',height=.52);b.scatter(vv,[y]*4,s=11,fc='white',ec='#243747',lw=.55,zorder=4);b.plot([min(vv),max(vv)],[y,y],c='#243747',lw=.8);b.text(total+.5,y,f'{total:.1f}',va='center',fontsize=8);ledger.append(dict(case=case,arm=arm,total_values=vv,mean_total=total,mean_schedule=s,mean_remainder=total-s,source_rows=rr))
 b.set(yticks=range(4),yticklabels=['P Native','P Adaptive','I Native','I Adaptive'],ylim=(3.6,-.6),xlim=(0,26.5),xlabel='Measured service interval (s)',title='(b) Recorded 4'+r'$\times$'+' wall ledger');clean(b,'x');fig.legend(handles=[Patch(fc=C['adaptive'],label='Scheduling'),Patch(fc='#D9E0E5',label='Remaining service')],ncol=2,loc='lower center',bbox_to_anchor=(.56,.005),frameon=False,fontsize=8)
 caption='Earlier headroom-development batch counts and wall-time accounting. Panel (a) is a vector matrix of arithmetic mean positive model-batch counts over four process repetitions; P/I and1/2/4 identify input and quota. A shared logarithmic blue scale encodes actual counts, annotated numerically. Panel (b) partitions each measured mean service interval at4-times quota into recorded scheduler time and its nonnegative remainder. Open markers and thin ranges retain all four total-time observations. These components are an accounting identity, not GPU/kernel timers or a causal decomposition. All per-process batch counts and interval components accompany the figure data.'
 save(fig,'fig_diagnosis_v17',2,1,caption,dict(batch_counts=counts,arms=arms,cells=cells,raw_services=rs,ledger=ledger),[path])


def confirmation():
 path='source_data/v17_verified/confirmation_all_primary_pairs.csv';pairs=get(path);cs=['fresh00_front','fresh00_interleave','fresh00_back','fresh01_front','fresh01_interleave','fresh01_back','alpaca_new00_interleave','alpaca_new01_interleave'];labs=['H1 F','H1 I','H1 B','H2 F','H2 I','H2 B','A1 I','A2 I'];fig,axs=plt.subplots(2,3,figsize=(DW,5.2),sharey=True);fig.subplots_adjust(left=.10,right=.985,bottom=.11,top=.90,wspace=.36,hspace=.50);data=[]
 for row,q in enumerate([1,4]):
  for col,metric in enumerate(['M_ratio','brief_ratio','batch_ratio']):
   a=axs[row,col];allv=[]
   for y,case in enumerate(cs):
    if y>=6:a.axhspan(y-.48,y+.48,color='#EDF1F3',zorder=-2)
    for ref,off in [('inc',-.17),('fixed',.17)]:
     rr=sorted([r for r in pairs if r['target']=='adaptive'and r['baseline']==ref and r['case']==case and int(r['quota_factor'])==q],key=lambda r:int(r['repeat']));assert len(rr)==4;v=[float(r[metric])if metric!='batch_ratio'else float(r['model_batches_target'])/float(r['model_batches_baseline'])for r in rr];allv+=v;s=strip(a,v,y+off,C['adaptive']if ref=='inc'else C['fixed'],mark='D'if ref=='inc'else'^',offset=.045);data.append(dict(case=case,quota=q,metric=metric,reference=ref,source_rows=rr,**s))
   a.set(xscale='log',xlim=(max(.01,min(allv)*.82),max(1.08,max(allv)*1.12)),ylim=(7.65,-.65),yticks=range(8),yticklabels=labs,xlabel='Adaptive / reference',title=f"({'abcdef'[row*3+col]}) "+['Cohort','Brief','Model batches'][col]+f', {q}'+r'$\times$');parity(a);clean(a,'x');logticks(a,'x',[.9,1,1.1,1.2,1.3]if metric=='brief_ratio'and q==4 else[.02,.05,.1,.2,.5,1,2])
 fig.legend(handles=[Line2D([],[],c=C['adaptive'],marker='D',ls='',ms=3.4,label='Reference: INC'),Line2D([],[],c=C['fixed'],marker='^',ls='',ms=3.4,label='Reference: Fixed INC-Q')],loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False)
 caption='Frozen eight-scenario adaptive confirmation. Rows separate base and fourfold quotas; the three substantive columns report cohort time, brief completion and positive model-batch counts, each relative to its same-block INC or Fixed INC-Q reference. All four process-pair observations are retained in every cell. Diamonds/triangles give geometric means; thick segments are empirical interquartile ranges and thin segments observed minima/maxima. H1/H2 identify document pairs, F/I/B arrivals and A1/A2 instruction cohorts; shading groups instruction rows. No confidence interval or global throughput bound is inferred from the batch-count comparison.'
 save(fig,'fig_confirmation_v17',2,3,caption,data,[path])


def width():
 path='source_data/v19_closeout/width/width_all_cuts.csv';rows=get(path);fig,axs=plt.subplots(2,1,figsize=(SW,4.75));fig.subplots_adjust(left=.23,right=.95,bottom=.11,top=.94,hspace=.53);data=[]
 for a,case,title in zip(axs,['fresh00_interleave','alpaca_new00_interleave'],['(a) Shared prefix','(b) Instructions']):
  rr=[r for r in rows if r['case']==case and r['quota_factor']=='1'and truth(r['ready_comparable'])];assert len(rr)in(697,1346);assert all(int(r['actual_positive_growth_width'])==int(r['omega_ready_one_page'])for r in rr);counts=collections.Counter((int(r['ready_count']),int(r['actual_positive_growth_width']))for r in rr)
  a.fill_between([0,4],[0,0],[0,4],color=C['fixed'],alpha=.055);a.plot([0,4],[0,4],ls='--',c=C['inc'],lw=.75)
  for (ready,actual),n in sorted(counts.items()):a.scatter(ready,actual,s=14+40*math.log10(n+1),fc=C['adaptive'],ec='white',lw=.5,zorder=3);a.annotate(str(n),(ready,actual),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8);data.append(dict(case=case,ready=ready,actual_positive_growth=actual,ready_bound=actual,cut_count=n))
  a.set(xlim=(-.4,4.4),ylim=(-.4,4.55),xticks=range(5),yticks=range(5),xlabel='Resident requests needing one page',ylabel='Funded one-page width',title=title);clean(a)
 caption='Ready demand and funded growth in the two base-quota final-deployment functional audits. The panels contain 697 and 1,346 comparable settled cuts, respectively. Marker coordinates count resident requests needing one new page and actual positive-growth width; the latter equals the ready-restricted fixed-witness bound at every plotted cut. Labels give exact cut counts and marker area is logarithmic in count. The dashed diagonal funds every ready request; the ochre region marks positive ready demand left unfunded, not a confidence region. Instruction cuts include 339 zero-width observations with positive free capacity. Cuts are dependent audit observations, not independent timing repeats or a global admission optimum.'
 save(fig,'fig_width_ready_v19',2,1,caption,dict(count_cells=data,all_comparable_source_rows=[r for r in rows if r['case']in['fresh00_interleave','alpaca_new00_interleave']and r['quota_factor']=='1'and truth(r['ready_comparable'])]),[path])


def projection():
 paths=['source_data/v19_projection/development/all_services.csv','source_data/v19_projection/confirmation/all_services.csv'];data=[];fig,axs=plt.subplots(2,3,figsize=(DW,4.8));fig.subplots_adjust(left=.12,right=.985,bottom=.12,top=.89,wspace=.46,hspace=.53)
 for row,(stage,path)in enumerate(zip(['Development','Fresh confirmation'],paths)):
  rs=[r for r in get(path)if r['phase']=='timing'];cases=sorted({r['case']for r in rs},key=lambda c:0 if('prefix'in c or c.startswith('fresh'))else 1);factors=sorted({int(r['quota_factor'])for r in rs});cells=[(c,q)for c in cases for q in factors];k={(r['case'],int(r['quota_factor']),int(r['repeat']),r['arm']):r for r in rs}
  for col,metric in enumerate(['M_s','brief_s','scheduler_ms']):
   a=axs[row,col];allv=[]
   for y,(case,q)in enumerate(cells):
    if y>=2:a.axhspan(y-.48,y+.48,color='#EDF1F3',zorder=-2)
    vals={}
    for arm,off in [('adaptive',-.18),('reuse',.18)]:
     rr=[k[(case,q,j,arm)]for j in range(6)];v=[1000*float(r['schedule_wall_s'])/int(r['schedule_calls'])if col==2 else float(r[metric])/float(k[(case,q,j,'native')][metric])for j,r in enumerate(rr)];vals[arm]=v;allv+=v;s=strip(a,v,y+off,C['inc']if arm=='adaptive'else C['adaptive'],mark='o'if arm=='adaptive'else'D',offset=.05);data.append(dict(stage=stage,case=case,quota=q,metric=metric,arm=arm,source_rows=rr,**s))
    for j in range(6):a.plot([vals['adaptive'][j],vals['reuse'][j]],[y-.18+(j-2.5)*.02,y+.18+(j-2.5)*.02],c='#A4B6C1',alpha=.45,lw=.45,zorder=1)
    if col==2:
     nr=[k[(case,q,j,'native')]for j in range(6)];nv=[1000*float(r['schedule_wall_s'])/int(r['schedule_calls'])for r in nr];allv+=nv;a.scatter(nv,[y]*6,c=C['native'],s=6,zorder=3);data.append(dict(stage=stage,case=case,quota=q,metric=metric,arm='native',values=nv,source_rows=nr,numerator='1000*schedule_wall_s',denominator='schedule_calls'))
   lo=min(allv)*.8;hi=max(allv)*1.2
   a.set(xscale='log',xlim=(min(lo,.94)if col<2 else lo,max(hi,1.06)if col<2 else hi),ylim=(3.6,-.6),yticks=range(4),yticklabels=[('P'if i<2 else'I')+str(q)+r'$\times$'for i,(c,q)in enumerate(cells)],xlabel='Time / Native'if col<2 else'Measured ms / call',title=f"({'abcdef'[row*3+col]}) "+['Cohort','Brief','Scheduling'][col]);clean(a,'x');a.xaxis.set_minor_formatter(NullFormatter())
   if col<2:parity(a);logticks(a,'x',[.5,1,1.5,2,3])
   else:logticks(a,'x',[.1,.3,1,3,10,30])
   if col==0:a.set_ylabel(stage)
 fig.legend(handles=[Line2D([],[],c=C['inc'],marker='o',ls='',ms=3.5,label='Reuse off'),Line2D([],[],c=C['adaptive'],marker='D',ls='',ms=3.5,label='Reuse on'),Line2D([],[],c=C['native'],ls='--',label='Native reference')],loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False)
 caption='Bounded projection reuse in two separate 7B timing populations. Rows are development and independently frozen fresh-input confirmation; columns give cohort time and brief completion relative to the same-block Native process, then recorded mean scheduling milliseconds per call. P/I denote prefix/instruction inputs and suffixes the measured quotas. Every cell retains all six matched process observations, with faint segments pairing reuse off/on. Large markers are geometric means, thick segments empirical interquartile ranges and thin segments observed ranges. Green points in the scheduling column are the six Native call-time averages. The scheduling ledger is not an Authority/ACK/kernel causal decomposition; stages are not pooled and no 14B reuse result is implied.'
 save(fig,'fig_projection_reuse_v19',2,3,caption,data,paths)


def main():
 old_replication();diagnostics();sensitivity();development();confirmation();width();projection()
 expected={'fig06_a800_confirmation','fig07_a800_native','fig08_a800_mechanism_alpaca','fig09_runtime_diagnostics','fig_sensitivity_v16','fig_quota_v17','fig_confirmation_v17','fig_width_ready_v19','fig_projection_reuse_v19','fig_diagnosis_v17'}
 assert {r['old_stem']for r in REG}==expected
 (D/'registry.json').write_text(json.dumps(REG,indent=2)+'\n');(D/'source_sha256.json').write_text(json.dumps(ALLSRC,indent=2)+'\n');(D/'CAPTIONS.txt').write_text('\n\n'.join(r['old_stem']+' -> '+r['new_stem']+'\n'+r['caption']for r in REG)+'\n')
 print(json.dumps(dict(status='TEN_MODERN_SI_DATA_BOUND_FIXED_WIDTH_EXPORTS_GENERATED_AWAITING_VISUAL_QA',figures=len(REG),new_GPU_services=0,source_files=len(ALLSRC)),indent=2))

if __name__=='__main__':main()
