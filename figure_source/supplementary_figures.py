"""Three current supplementary figures from unchanged measured tables."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,os,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
from matplotlib.ticker import NullFormatter
W=Path(__file__).resolve().parents[1];O=W/'results/figure_build/figures';D=W/'results/figure_build/supplementary'
SW=3.16697799916978;DW=6.500000276740003
C={'inc':'#687E88','fixed':'#A37232','adaptive':'#285F8C','native':'#28776F','banker_c0':'#A37232','hard_reserve':'#687E88','reuse':'#285F8C'}
ALLSRC={};REG=[];EXPECTED={};FONT_FAMILY='Times New Roman'
def setup():
    global FONT_FAMILY
    extra=os.environ.get('TSMC_FONT_DIR')
    if extra:
        for p in Path(extra).glob('*.ttf'):font_manager.fontManager.addfont(str(p))
    available={f.name for f in font_manager.fontManager.ttflist}
    preferred=os.environ.get('TSMC_FONT_FAMILY','Times New Roman')
    FONT_FAMILY=preferred if preferred in available else 'STIXGeneral'
    if FONT_FAMILY not in available or FONT_FAMILY not in {'Times New Roman','STIXGeneral','Liberation Serif'}:
        raise RuntimeError('Install a licensed Times New Roman font or the compatible STIXGeneral/Liberation Serif font before rebuilding.')
    plt.rcParams.update({'font.family':FONT_FAMILY,'mathtext.fontset':'cm','font.size':9,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'lines.linewidth':.9})

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
 assert data==EXPECTED[old], f'Frozen displayed endpoints differ: {old}'
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
 REG.append(dict(new_stem=stem,rows=rows,cols=cols,span=span,width_in=width,height_in=height,minimum_font_pt=min(texts),caption=caption,plot_data=str(dp.relative_to(W)),plot_data_path=str(dp.relative_to(W)),plot_data_sha256=sha(dp),source_sha256=source_hash,outputs=files,figure_sha256={k:v['sha256']for k,v in files.items()},recipe_sha256=sha(Path(__file__)),frozen_endpoints_sha256=sha(W/'source_data/plot_endpoints/supplementary.json'),font_family=FONT_FAMILY,publication_typography=FONT_FAMILY=='Times New Roman',fixed_geometry=True,vector_PDF=True,live_text_SVG=True,native_PPT_claim=False))

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
 caption='Admission sensitivity on the two exposed inputs. Rows separate prefix and instruction cohorts; columns report cohort time relative to same-block Native, brief completion relative to Native, and actual mean requests per positive model batch. All three process repetitions at each measured gate values 1/2/3/4/6/8 are shown. Ratio centers are geometric means and batch-width centers arithmetic means. Curves connect measured centers only; shading is the observed minimum-to-maximum process range, not a confidence band. Ochre references are the measured gate-off deployment, with its observed range. Dashed equality at 1 applies only to the two ratio columns. No unmeasured parameter optimum or workload-population inference is implied.'
 save(fig,'fig_sensitivity_v16',2,3,caption,data,[path,'source_data/v16_verified/w52_final.json'])

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
 global W,O,D,EXPECTED
 parser=argparse.ArgumentParser(description='Rebuild the three current TSMC supplementary figures.')
 parser.add_argument('--manuscript-root',type=Path,default=W)
 W=parser.parse_args().manuscript_root.resolve();O=W/'results/figure_build/figures';D=W/'results/figure_build/supplementary';O.mkdir(exist_ok=True);D.mkdir(parents=True,exist_ok=True)
 EXPECTED=json.loads((W/'source_data/plot_endpoints/supplementary.json').read_text())
 setup();sensitivity();width();projection()
 assert len(REG)==3
 for number,row in enumerate(REG,1):row.update(document='supplementary',figure_number=f'S{number}')
 (D/'registry.json').write_text(json.dumps({'release':'1.2.0','figures':REG,'source_hashes':ALLSRC,'all_displayed_endpoints_equal_frozen':True},indent=2)+'\n')
 print(json.dumps({'status':'PASS','figures':3,'font_family':FONT_FAMILY,'publication_typography':FONT_FAMILY=='Times New Roman'}))
if __name__=='__main__':main()
