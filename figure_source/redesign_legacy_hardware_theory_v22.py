"""Two supporting figures; archived inputs only. No GPU execution.
Fixed physical export; live SVG text and embedded TrueType PDF fonts.
"""
from pathlib import Path
import csv, json, hashlib, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from matplotlib.patches import Rectangle

SINGLE = 3.16697799916978
DOUBLE = 6.500000276740003
BLUE, GREEN, GOLD, GRAY = '#285F8C', '#28776F', '#A37232', '#687E88'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def configure():
    for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm',
        'font.size':9,'axes.labelsize':9,'axes.titlesize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.bbox':None,'savefig.pad_inches':0})

def export(fig, w, old, columns, rows, height, caption, data, sources, output):
    stem=old+'_v22'
    paths=[]
    for suffix in ['pdf','svg','png']:
        p=w/'figs'/f'{stem}.{suffix}'
        fig.savefig(p,dpi=240)
        paths.append(str(p.relative_to(w)))
    fig.canvas.draw()
    # Receipt includes actual text sizes and axes; no cropping/resizing is used.
    texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_text()]
    minimum=min(t.get_fontsize() for t in texts)
    assert minimum>=8.0, (old,minimum)
    plt.close(fig)
    receipt={'old_stem':old,'new_stem':stem,'panel_columns':columns,'panel_rows':rows,
        'width_inches':DOUBLE if columns==3 else SINGLE,'height_inches':height,
        'span':'double' if columns==3 else 'single','minimum_font_pt':minimum,
        'caption':caption,'assets':paths,'fonts':'Times New Roman; math Computer Modern',
        'intervals':'Observed sample SD or archived setting min/max, as specifically labelled; no CI.',
        'data_scope':data['scope']}
    (output/f'{old}_plotrows.json').write_text(json.dumps(data,indent=2)+'\n')
    return receipt

def generate(w=None, out=None, record=None):
    w=Path(w or Path(__file__).resolve().parents[1])
    out=Path(out or w/'source_data/v22_figures/legacy_hardware_theory')
    out.mkdir(parents=True,exist_ok=True)
    configure()
    sources=[]; registry=[]
    hp=w/'source_data/figure_inputs/fig02_hardware.json'
    h=json.loads(hp.read_text());sources.append({'path':str(hp.relative_to(w)),'sha256':sha(hp)})
    hw=['local_5090','a100_40gb','a800_80gb','rtx6000_pro_96gb']
    names=['RTX 5090','A100 40 GB','A800 80 GB','RTX PRO 6000\n96 GB']
    mods=['7b','14b','32b']
    z=np.array([[h['median_ms'][m][d] if h['median_ms'][m][d] is not None else np.nan for m in mods]for d in hw])
    fig=plt.figure(figsize=(SINGLE,2.68)); a=fig.add_axes([.295,.285,.61,.57]); cax=fig.add_axes([.295,.135,.61,.04])
    cmap=colors.LinearSegmentedColormap.from_list('archive_latency',['#F0F5F7','#B1C5CE',BLUE])
    mesh=a.pcolormesh(np.arange(4),np.arange(5),np.ma.masked_invalid(z),cmap=cmap,vmin=300,vmax=1300,edgecolors='white',linewidth=1.5)
    rr=[]
    for i,d in enumerate(hw):
        for j,m in enumerate(mods):
            v=h['median_ms'][m][d]; n=h['request_record_counts'][m][d]
            rr.append({'hardware':d,'model':m,'median_backend_latency_ms':v,'completed_request_records':n,'missing':v is None})
            if v is None:
                a.add_patch(Rectangle((j,i),1,1,facecolor='#F5F5F5',edgecolor='#B8C0C4',hatch='////',linewidth=.5))
                s='Not\nrecorded'; col='#3E4549'
            else:s=f'{v:.0f} ms\n$n$={int(n):,}';col='white' if v>=900 else '#182D3B'
            a.text(j+.5,i+.5,s,ha='center',va='center',fontsize=8,color=col,linespacing=1.35)
    a.set_xticks(np.arange(3)+.5,['7B','14B','32B']);a.set_yticks(np.arange(4)+.5,names)
    a.set_ylim(4,0);a.tick_params(length=0,pad=4)
    a.set_title('Archived backend latency',loc='left',pad=5,fontweight='bold')
    cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[300,800,1300]);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.ax.tick_params(labelsize=8,length=2)
    cb.set_label('Completed-request median (ms)',labelpad=2,fontsize=8)
    fig.text(.295,.965,'Separate hosts / configurations',ha='left',va='top',fontsize=8,color=GRAY)
    cap=('Archived backend-latency proxies on separate hosts and configurations. Rows identify GPUs and columns model sizes; vector cells show completed-request medians in milliseconds and record counts n. A single global color scale is used. Hatched cells mean not recorded, not zero or failure. These are neither streaming TTFT/TPOT measurements nor a matched heterogeneous-serving comparison. The repeated scaling view is omitted without removing any measured cell.')
    registry.append(export(fig,w,'fig02_hardware',1,1,2.68,cap,{'scope':h['scope'],'rows':rr},sources,out))

    qp=w/'source_data/fig03_theory.json'; cp=w/'experiments/routing/results/controlled_runs.csv'
    q=json.loads(qp.read_text())['plotted_series']['reserve']; raw=list(csv.DictReader(cp.open()))
    sources.extend({'path':str(p.relative_to(w)),'sha256':sha(p)} for p in [qp,cp])
    # All six methods are retained in plot data. Verify exact coincident groups
    # before showing one raw point series for each group, without invented offsets.
    selected=[r for r in raw if float(r['alpha'])==.025]
    by={(r['method'],float(r['label_noise']),int(r['T']),int(r['seed'])):r for r in selected}
    groups={'Restricted / no residual':['Queue','Locality','Flat collapsed','Typed no residual'],
            'Typed / sufficient flat':['TypedRoute','Flat sufficient']}
    for methods in groups.values():
        for (m,n,t,s),r in by.items():
            if m in methods:
                assert float(r['cumulative_loss'])==float(by[(methods[0],n,t,s)]['cumulative_loss'])
    fig,aa=plt.subplots(1,3,figsize=(DOUBLE,3.00),gridspec_kw={'width_ratios':[1,1,1]})
    fig.subplots_adjust(left=.075,right=.99,bottom=.23,top=.80,wspace=.48)
    cols=[BLUE,GREEN,GOLD,GRAY]; arith=[]
    for k,(name,vals) in enumerate(q.items()):
        x=np.array([r['N_samples']for r in vals]);mid=np.array([100*r['median']for r in vals]);low=np.array([100*r['min']for r in vals]);high=np.array([100*r['max']for r in vals])
        aa[0].fill_between(x,low,high,color=cols[k],alpha=.13,lw=0)
        aa[0].plot(x,mid,color=cols[k],marker=['o','s','^','D'][k],markersize=2.4,lw=.9,label=name,ls=['-','--','-.',':'][k])
        arith.extend({'series':name,**v,'display_percent':100*v['median']}for v in vals)
    aa[0].set_xscale('log');aa[0].set_ylim(-2,68);aa[0].set_xticks([10,100,1000],['10','100','1k'])
    aa[0].set_xlabel('Nominal sample size $N$');aa[0].set_ylabel('Reserve reduction (%)')
    aa[0].legend(ncol=2,loc='upper left',frameon=False,columnspacing=.5,handlelength=1.1,handletextpad=.35,borderaxespad=.2)
    aa[0].set_title('(a) Bound arithmetic',loc='left',fontweight='bold',pad=7)
    summaries=[]; plotted=[]
    for panel,key,xvals in [(1,'T',[100,500,1000,5000,10000]),(2,'noise',[0,.05,.1,.2,.4])]:
        a=aa[panel]
        for gi,(label,methods)in enumerate(groups.items()):
            method=methods[0]; xx=[];means=[];sds=[]
            for x in xvals:
                filt=[r for r in selected if r['method']==method and ((float(r['label_noise'])==0 and int(r['T'])==x)if key=='T'else(float(r['label_noise'])==x and int(r['T'])==10000))]
                assert len(filt)==20
                yy=np.array([float(r['cumulative_loss']if key=='T'else r['per_request_loss'])for r in filt]);yy=yy if key=='T'else yy/.025
                xx.append(x);means.append(yy.mean());sds.append(yy.std(ddof=1))
                a.scatter(np.full(20,x),yy,s=8,facecolor='none',edgecolor=[GOLD,BLUE][gi],linewidth=.45,alpha=.36,zorder=2)
                summaries.append({'panel':panel,'group':label,'methods':methods,'x':x,'n_seeds':20,'mean':float(yy.mean()),'sample_SD':float(yy.std(ddof=1))})
                plotted.extend({'panel':panel,**r}for r in selected if r['method']in methods and ((float(r['label_noise'])==0 and int(r['T'])==x)if key=='T'else(float(r['label_noise'])==x and int(r['T'])==10000)))
            a.errorbar(xx,means,yerr=sds,marker=['o','s'][gi],markersize=3,lw=1,color=[GOLD,BLUE][gi],ls=['--','-'][gi],capsize=2,zorder=3,label=label)
        if key=='T':
            a.plot(xvals,.025*np.array(xvals)/2,color='#333333',ls=':',lw=.9)
            a.set_xlim(-300,10500);a.set_xticks([0,5000,10000],['0','5k','10k']);a.set_ylim(-7,145)
            a.set_xlabel('Routing requests $T$');a.set_ylabel('Cumulative mismatch loss')
            a.text(.04,.85,r'Dotted: $\alpha T/2$',transform=a.transAxes,fontsize=8,color='#333333')
            a.set_title('(b) Executed policies',loc='left',fontweight='bold',pad=7)
        else:
            a.set_ylim(-.04,.61);a.set_xticks([0,.2,.4]);a.set_xlabel('Type-label flip probability');a.set_ylabel(r'Mean loss / $\alpha$')
            a.set_title('(c) Label-noise control',loc='left',fontweight='bold',pad=7)
    for a in aa:
        a.grid(axis='y',color='#DCE3E7',lw=.45,zorder=0);a.set_axisbelow(True)
    fig.text(.075,.965,'CPU calculations and controlled policy execution; not GPU serving measurements',va='top',fontsize=9,fontweight='bold')
    handles,labels=aa[1].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.66,.005),ncol=2,frameon=False,columnspacing=1.1,handlelength=1.6)
    fig.text(.075,.012,'Bands: setting range\nPoints: 20 seeds; bars: sample SD',va='bottom',fontsize=8,color=GRAY)
    cap=('Supporting CPU calculations and controlled policy execution, not GPU-serving benchmarks. (a) Median reserve-reduction arithmetic across archived settings; shaded bands show setting minima and maxima, not confidence intervals. (b) Actually executed routing at alpha=0.025 and zero label noise, with the expectation alpha*T/2 as a dotted reference. (c) Label-noise control at T=10,000. Panels (b,c) retain all 20 seed observations as open points and display means with sample SD. Queue, Locality, Flat collapsed and Typed no residual coincide exactly; TypedRoute and Flat sufficient coincide exactly. Coincident methods are grouped only after checking every corresponding archived record. No curve is a live heterogeneous-GPU result.')
    registry.append(export(fig,w,'fig03_theory_executed',3,1,3.00,cap,{'scope':'Controlled CPU arithmetic and executed routing; no GPU measurement. Identical method groups verified recordwise.','reserve_rows':arith,'routing_raw_rows':plotted,'routing_summaries':summaries,'alpha':.025,'seed_count_per_setting':20,'groups':groups},sources,out))
    (out/'sourcehash.json').write_text(json.dumps({'inputs':sources,'recipe_sha256':sha(Path(__file__))},indent=2)+'\n')
    (out/'layoutcaptions.json').write_text(json.dumps(registry,indent=2)+'\n')
    if record is not None:
        record.extend(registry)
    return registry

if __name__=='__main__':
    print(json.dumps(generate(),indent=2))
