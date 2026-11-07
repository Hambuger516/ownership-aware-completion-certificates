#!/usr/bin/env python3
"""Print-size, vector-only experimental figures from immutable measured endpoints.

This recipe neither runs experiments nor edits any original measurement or plot.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter
import numpy as np
from pypdf import PdfReader

W=Path(__file__).resolve().parents[1]
OUT=W/'results/figure_build/main'
FIGS=W/'results/figure_build/figures'
FONT_FAMILY='Times New Roman'
HEIGHT_SCALE=1.0
WIDTH=3.16697799916978
FONT=10.0
BLUE='#285F8C'; TEAL='#28776F'; GRAY='#687E88'; OCHRE='#A37232'
COLORS=['#3D789E','#EEF5F8','#FFFFFF','#F6EDDE','#B78342']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gm=lambda v:math.exp(statistics.mean(math.log(x) for x in v))

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def study(root,kind):
    base=root/'source_data/v18_verified'/kind
    data=base/'independent_readout_v1' if (base/'independent_readout_v1').exists() else base
    j=json.loads((data/'FINAL_NUMERIC_READOUT.json').read_text())
    assert j['status']=='COMPLETE_RAW_SOURCE_RESOURCE_MODEL_OUTPUT_WORK_REPLAY_PASS'
    assert j['all_formal_outputs_exact'] and j['all_controller_reexecution_zero']
    return base,data

def compute_matrix(root,kind,parent):
    base,data=study(root,kind);protocol=json.loads((base/'PROTOCOL.json').read_text())
    cases=protocol['cases'];quotas=protocol['factors'];arms=protocol['arms']
    service=[r for r in read(data/'all_services.csv')if r['phase']=='timing']
    key={(r['case'],int(r['quota_factor']),int(r['repeat']),r['arm']):r for r in service}
    assert len(key)==len(cases)*len(quotas)*4*len(arms)
    source_pairs=read(data/'all_pairs.csv')
    for pair in source_pairs:
        k=(pair['case'],int(pair['quota_factor']),int(pair['repeat']))
        for metric,field in [('M_s','M_ratio'),('brief_s','brief_ratio')]:
            ratio=float(key[k+(pair['target'],)][metric])/float(key[k+(pair['baseline'],)][metric])
            assert math.isclose(ratio,float(pair[field]),rel_tol=1e-12,abs_tol=1e-12)
    contrasts=([('adaptive','banker_c0','Banker C0'),('adaptive','hard_reserve','Hard res.'),('adaptive','native','Native')]
               if kind=='baselines' else [('inc','native','INC'),('fixed','native','Fixed'),('adaptive','native','Adaptive')])
    groups=[]
    for row,metric in enumerate(['M_s','brief_s']):
        for col,case in enumerate(cases):
            for method,(target,baseline,label)in enumerate(contrasts):
                for quota in quotas:
                    pairs=[(key[(case,quota,rep,target)],key[(case,quota,rep,baseline)])for rep in range(4)]
                    assert all(a['host']==b['host']and a['GPU']==b['GPU']for a,b in pairs)
                    values=[float(a[metric])/float(b[metric])for a,b in pairs]
                    groups.append(dict(study=kind,case=case,metric=metric,quota=quota,target=target,
                        baseline=baseline,row_label=label,panel_row=row,panel_column=col,method_row=method,
                        raw_ratios=values,repeat=list(range(4)),GM=gm(values),minimum=min(values),maximum=max(values)))
    parent_stem='fig_fair_baselines_v30'if kind=='baselines'else'fig_scale14b_v30'
    assert groups==parent['displayed'][parent_stem], 'The independently recomputed process/statistic records differ from the frozen displayed endpoints'
    sources=[data/name for name in ['all_services.csv','all_pairs.csv','FINAL_NUMERIC_READOUT.json']]+[base/'PROTOCOL.json']
    return groups,quotas,contrasts,sources,len(source_pairs)

def setup():
    global FONT_FAMILY,HEIGHT_SCALE
    extra=os.environ.get('TSMC_FONT_DIR')
    if extra:
        for p in Path(extra).glob('*.ttf'):font_manager.fontManager.addfont(str(p))
    available={f.name for f in font_manager.fontManager.ttflist}
    preferred=os.environ.get('TSMC_FONT_FAMILY','Times New Roman')
    FONT_FAMILY=preferred if preferred in available else 'STIXGeneral'
    if FONT_FAMILY not in available or FONT_FAMILY not in {'Times New Roman','STIXGeneral','Liberation Serif'}:
        raise RuntimeError('Install a licensed Times New Roman font or the compatible STIXGeneral/Liberation Serif font before rebuilding.')
    HEIGHT_SCALE=1.0 if FONT_FAMILY=='Times New Roman' else 1.3
    plt.rcParams.update({'font.family':FONT_FAMILY,'mathtext.fontset':'cm','font.size':FONT,
        'axes.titlesize':FONT,'axes.labelsize':FONT,'xtick.labelsize':FONT,'ytick.labelsize':FONT,
        'legend.fontsize':FONT,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,
        'xtick.major.width':.55,'ytick.major.width':.55,'savefig.bbox':None})

def export(fig,name,old,rows,cols,caption,caption_tex,sources,records,encoding,extra):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
    texts=[a for a in fig.findobj(matplotlib.text.Text) if a.get_visible() and a.get_text().strip()]
    assert min(a.get_fontsize() for a in texts)>=10
    assert math.isclose(fig.get_figwidth(),WIDTH,abs_tol=1e-12)
    for a in texts:
        b=a.get_window_extent(renderer)
        assert b.x0>=canvas.x0-.5 and b.y0>=canvas.y0-.5 and b.x1<=canvas.x1+.5 and b.y1<=canvas.y1+.5,a.get_text()
    overlap=[]
    for i,a in enumerate(texts):
        b=a.get_window_extent(renderer)
        for c in texts[i+1:]:
            d=c.get_window_extent(renderer)
            if a.get_text()==c.get_text() and b.bounds==d.bounds:continue
            ox=min(b.x1,d.x1)-max(b.x0,d.x0);oy=min(b.y1,d.y1)-max(b.y0,d.y0)
            if ox>1 and oy>1:overlap.append([a.get_text(),c.get_text(),ox,oy])
    assert not overlap,overlap
    artist_counts={'texts':len(texts),'axes':len(fig.axes),'lines':sum(len(a.lines) for a in fig.axes),
        'patches':sum(len(a.patches) for a in fig.axes),'collections':sum(len(a.collections) for a in fig.axes)}
    size=[fig.get_figwidth(),fig.get_figheight()]
    for ext in ['pdf','svg','png']:
        fig.savefig(FIGS/f'{name}.{ext}',dpi=300,bbox_inches=None,pad_inches=0,
            metadata={'Creator':'TSMC current-figure reproduction'} if ext=='pdf' else None)
    plt.close(fig)
    page=PdfReader(FIGS/f'{name}.pdf').pages[0]
    assert abs(float(page.mediabox.width)-72*WIDTH)<.001
    fonts=[]
    for val in page['/Resources']['/Font'].values():
        f=val.get_object();assert f['/Subtype']!='/Type3';d=f.get('/FontDescriptor')
        if '/DescendantFonts' in f:d=f['/DescendantFonts'][0].get_object().get('/FontDescriptor')
        d=d.get_object();assert any(k in d for k in ['/FontFile','/FontFile2','/FontFile3']);fonts.append(str(f['/BaseFont']))
    xo=page['/Resources'].get('/XObject',{});images=0
    if xo:images=sum(v.get_object().get('/Subtype')=='/Image' for v in xo.get_object().values())
    assert images==0
    svg=(FIGS/f'{name}.svg').read_text();assert '<text' in svg and FONT_FAMILY in svg
    return {'new_stem':name,'rows':rows,'cols':cols,'span':'single','width_in':size[0],
        'height_in':size[1],'font_family':FONT_FAMILY,'publication_typography':FONT_FAMILY=='Times New Roman','fallback_height_multiplier':HEIGHT_SCALE,'math_fontset':'Computer Modern',
        'minimum_font_pt':min(a.get_fontsize() for a in texts),'preferred_font_pt':FONT,
        'manuscript_linear_scale':.8,'printed_minimum_font_pt':.8*min(a.get_fontsize() for a in texts),
        'printed_width_in':.8*size[0],'printed_height_in':.8*size[1],
        'caption_plain':caption,'caption_tex':caption_tex,'encoding':encoding,
        'source_sha256':{str(p.relative_to(W)):sha(p) for p in sources},
        'code_sha256':sha(Path(__file__)),'frozen_endpoints_sha256':sha(W/'source_data/plot_endpoints/main.json'),
        'figure_sha256':{e:sha(FIGS/f'{name}.{e}') for e in ['pdf','svg','png']},
        'plot_data_path':'results/figure_build/main/plotrecords.json','plot_group_count':len(records),
        'artist_counts':artist_counts,'pdf_fonts_embedded':fonts,'pdf_type3_fonts':False,
        'pdf_image_xobjects':images,'svg_live_text_count':svg.count('<text'),
        'all_visible_text_within_canvas':True,'visible_text_bbox_overlaps':0,'visual_signoff':False,**extra}

def ownership(parent):
    _,d=study(W,'ownership');raw=read(d/'all_pairs.csv');assert len(raw)==36
    services=[r for r in read(d/'all_services.csv') if r['phase']=='timing'];assert len(services)==72
    key={(r['case'],int(r['repeat']),r['arm']):r for r in services}
    for r in raw:
        for field,metric in [('M_ratio','M_s'),('brief_ratio','brief_s')]:
            k=(r['case'],int(r['repeat']))
            assert math.isclose(float(key[k+(r['target'],)][metric])/float(key[k+(r['baseline'],)][metric]),float(r[field]),rel_tol=1e-12,abs_tol=1e-12)
    cases=[f'confirm{i:02}_{p}' for i in range(3) for p in ['front','interleave','back']]
    labels=[f'P{i+1} {p}' for i in range(3) for p in ['F','I','B']]+['All']
    fig,axes=plt.subplots(1,2,figsize=(WIDTH,3.5*HEIGHT_SCALE));fig.subplots_adjust(left=.185,right=.97,bottom=.19,top=.815,wspace=.31)
    records=[];overall=[]
    for col,metric in enumerate(['M_ratio','brief_ratio']):
        ax=axes[col]
        for group in [0,2]:ax.axhspan(group*3-.48,group*3+2.48,color='#F2F4F5',zorder=0)
        for y in [1,4,7]:ax.axhspan(y-.43,y+.43,color='#E6EFF6',zorder=0)
        ax.axhspan(8.64,9.5,color='#E9ECEF',zorder=0)
        ax.axvline(1,color='#565656',lw=.7,ls=(0,(3,2)),zorder=1)
        for y,case in enumerate(cases):
            rr=sorted([r for r in raw if r['case']==case],key=lambda r:int(r['repeat']))
            assert len(rr)==4 and [int(r['repeat']) for r in rr]==list(range(4))
            assert all(r['target']=='full_c1' and r['baseline']=='full_c0' for r in rr)
            v=[float(r[metric]) for r in rr];color=BLUE if case.endswith('interleave') else GRAY
            ax.plot([min(v),max(v)],[y,y],color=color,lw=1.3,zorder=3)
            ax.scatter(v,[y+(i-1.5)*.105 for i in range(4)],s=9,facecolors='white',edgecolors=color,lw=.65,zorder=4)
            ax.plot(gm(v),y,'D',color=color,ms=3.4,zorder=5)
            records.append(dict(study='ownership',case=case,metric=metric,repeat=list(range(4)),raw_ratios=v,GM=gm(v),minimum=min(v),maximum=max(v)))
        v=[float(r[metric]) for r in raw];g=gm(v)
        ax.plot(g,9,'D',ms=5.7,color=BLUE,mec='white',mew=.5,zorder=5)
        ax.text(.5,-.15,f'Overall GM {g:.3f}',transform=ax.transAxes,ha='center',fontsize=10,color=BLUE)
        overall.append({'metric':metric,'paired_process_blocks':36,'fixed_scenarios':9,'GM':g,'displayed_GM':f'{g:.3f}',
            'summary_has_range':False,'definition':'geometric mean over all 36 matched process ratios; scenarios equally represented'})
        ax.set(xscale='log',ylim=(9.55,-.55),yticks=range(10));ax.set_yticklabels(labels if col==0 else [])
        ax.tick_params(axis='y',length=0,pad=4)
        if col==0:ax.set_xlim(.70,1.065);ax.set_xticks([.7,.8,.9,1],labels=['0.7','0.8','0.9','1'])
        else:ax.set_xlim(.045,1.12);ax.set_xticks([.05,.2,1],labels=['0.05','0.2','1'])
        ax.xaxis.set_minor_formatter(NullFormatter());ax.tick_params(which='minor',length=0)
        ax.grid(axis='x',color='#DCE1E4',lw=.4,zorder=0);ax.set_title(['(a) Cohort $M$','(b) Brief $B$'][col],pad=7)
    assert records==parent['displayed']['fig_ownership_repeated_v30']
    fig.text(.57,.023,'Full C1 / Full C0 (log scale)',ha='center',fontsize=FONT)
    fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=GRAY,ms=3.2,ls='',label='Paired runs'),
        Line2D([],[],marker='D',color=BLUE,ms=3.4,ls='',label='GM')],loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,
        frameon=False,columnspacing=.85,handletextpad=.35)
    fig.text(.57,.89,'9 scenarios; 4 matched pairs each',ha='center',fontsize=10,color='#42525A')
    cap=('Ownership ablation: Full C1/Full C0 cohort and brief completion ratios for three input pairs '
         'under front (F), interleaved (I), and back (B) arrivals. Open points are matched repeats, '
         'segments are observed ranges, and diamonds are geometric means. Blue bands mark interleaving; '
         'the All row summarizes all 36 pairs. Dashed lines mark equality.')
    reg=export(fig,'fig_ownership_repeated_v30','fig_ownership_repeated_v22',1,2,cap,cap,
        [d/n for n in ['all_pairs.csv','all_services.csv','FINAL_NUMERIC_READOUT.json']],records,
        'Paired observed-range forest; arrival-category bands, input-group shading, and 36-pair overall summary diamond',
        {'formal_services':72,'observed_paired_ratios_per_endpoint':36,'raw_displayed_points':72,
         'observed_range_segments':18,'scenario_GM_diamonds':18,'overall_summary_diamonds':2,'summary_records':overall})
    return reg,records,overall

def matrix(kind,parent):
    groups,quotas,contrasts,sources,npairs=compute_matrix(W,kind,parent)
    base=(kind=='baselines');name='fig_fair_baselines_v30' if base else 'fig_scale14b_v30'
    old='fig_fair_baselines_v24' if base else 'fig_scale14b_v24'
    fig=plt.figure(figsize=(WIDTH,3.25*HEIGHT_SCALE));grid=fig.add_gridspec(2,2,left=.25,right=.985,bottom=.25,top=.815,hspace=.73,wspace=.16)
    norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=2);cmap=LinearSegmentedColormap.from_list('benefit_cost_v30',COLORS)
    outline_count=0
    for row in range(2):
        for col in range(2):
            ax=fig.add_subplot(grid[row,col]);rs=[r for r in groups if r['panel_row']==row and r['panel_column']==col]
            vals=np.array([[math.log2(next(r['GM'] for r in rs if r['method_row']==m and r['quota']==q)) for q in quotas] for m in range(3)])
            assert np.all(vals>=-6) and np.all(vals<=2)
            ax.pcolormesh(np.arange(len(quotas)+1),np.arange(4),vals,cmap=cmap,norm=norm,shading='flat',edgecolors='#CAD3D9',linewidth=.55,rasterized=False)
            for r in rs:
                x=quotas.index(r['quota']);y=r['method_row'];rgb=cmap(norm(math.log2(r['GM'])))[:3]
                linear=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in rgb]
                lum=sum(c*w for c,w in zip(linear,[.2126,.7152,.0722]));ink='white' if lum<.23 else '#172F3E'
                ax.text(x+.5,y+.5,f"{r['GM']:.3f}",ha='center',va='center',fontsize=FONT,color=ink,
                    fontweight='bold' if y==2 else 'normal')
                if base and row==0 and col==1 and r['baseline']=='hard_reserve' and r['quota']==4:
                    ax.add_patch(Rectangle((x+.018,y+.025),.964,.95,fill=False,edgecolor=OCHRE,linewidth=1.0,zorder=6));outline_count+=1
            ax.add_patch(Rectangle((.012,2.018),len(quotas)-.024,.964,fill=False,edgecolor=BLUE,linewidth=1.05,zorder=5))
            assert not ax.lines and not any(isinstance(c,matplotlib.collections.PathCollection) for c in ax.collections)
            ax.set_xlim(0,len(quotas));ax.set_ylim(3,0)
            ax.set_xticks(np.arange(len(quotas))+.5,labels=[str(q) for q in quotas])
            ax.set_yticks([.5,1.5,2.5],labels=[r[2] for r in contrasts] if col==0 else [])
            ax.tick_params(length=0,pad=3)
            for s in ax.spines.values():s.set_visible(False)
            ax.set_title(f'({"abcd"[row*2+col]}) '+('Prefix' if col==0 else 'Instr.')+(' $M$' if row==0 else ' $B$'),pad=5)
            if row==1:ax.set_xlabel('Quota / base',labelpad=3)
    cax=fig.add_axes([.29,.105,.63,.025]);sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    cb=fig.colorbar(sm,cax=cax,orientation='horizontal',ticks=[-6,0,2]);cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
    cb.ax.set_xticklabels(['1/64','1','4']);cb.ax.tick_params(length=2,pad=1);cb.outline.set_linewidth(.4)
    fig.text(.425,.045,'Faster',ha='center',color=BLUE,fontsize=10)
    fig.text(.815,.045,'Slower',ha='center',color=OCHRE,fontsize=10)
    fig.text(.60,.965,'Adaptive / comparator' if base else 'Method / Native',ha='center')
    fig.text(.60,.895,'4 matched pairs per cell',ha='center',fontsize=10,color='#42525A')
    if base:
        cap=('7B Adaptive/comparator ratios for cohort time $M$ and brief completion $B$. '
             'Each cell is a geometric mean of four matched pairs. Blue frames identify Native comparisons; '
             'the ochre frame marks the instruction $4\\times$ hard-reservation cohort result. '
             'Colours encode ratios below/above equality using the shared $(1/64,1,4)$ scale.')
    else:
        cap=('14B method/Native ratios for cohort time $M$ and brief completion $B$, with four matched pairs per cell. '
             'Blue frames identify Adaptive. The equality-centred colour encoding and range $(1/64,1,4)$ match Fig.~\\ref{fig:quota-main}.')
    plain=cap.replace('$','').replace('\\times','x').replace('Fig.~\\ref{fig:quota-main}','the 7B matrix')
    reg=export(fig,name,old,2,2,plain,cap,sources,groups,
        'Mean-only ratio matrices; common equality-centred log2 colour scale, selected-row frames and retained unfavorable cell',
        {'formal_services':96 if base else 64,'audit_services':24 if base else 16,'cell_count':len(groups),
         'matched_pairs_per_cell':4,'shared_log2_ratio_limits':[-6,0,2],'shared_ratio_limits':[1/64,1,4],
         'cmap_colours':COLORS,'main_raw_points':0,'main_range_segments':0,'main_dotted_cell_markers':0,
         'main_cell_gm_font_pt':FONT,'selected_row_frames':4,'unfavorable_cell_frames':outline_count,
         'normalization':'TwoSlopeNorm(log2(ratio),vmin=-6,vcenter=0,vmax=2)',
         'all_groups_match_frozen_displayed_endpoints':True,'all_pair_ratios_recomputed_from_services':True})
    return reg,groups

def runtime(parent):
    d=W/'source_data/runtime_accounting';raw=read(d/'runtime_factor4_per_service.csv')
    ledger=json.loads((d/'RUNTIME_ACCOUNTING.json').read_text())
    assert ledger['formal_services']==72 and ledger['all_existing_control_cut_timing_sums_match']
    raw=[r for r in raw if r['stage']=='confirmation'];assert len(raw)==36
    records=[];derived=[];fig=plt.figure(figsize=(WIDTH,3.75*HEIGHT_SCALE))
    outer=fig.add_gridspec(2,1,left=.235,right=.975,bottom=.13,top=.855,hspace=.79)
    for index,(family,title) in enumerate([('prefix','(a) Shared prefix'),('instruction','(b) Instructions')]):
        inner=outer[index].subgridspec(1,2,width_ratios=[1.48,1],wspace=.20)
        ax=fig.add_subplot(inner[0]);cost=fig.add_subplot(inner[1],sharey=ax)
        local=[]
        for y,arm in zip([2,1,0],['native','adaptive','reuse']):
            rr=sorted([r for r in raw if r['family']==family and r['arm']==arm],key=lambda r:int(r['repeat']))
            assert len(rr)==6 and [int(r['repeat']) for r in rr]==list(range(6))
            m=[float(r['M_s']) for r in rr];s=[float(r['schedule_wall_s']) for r in rr]
            pc=[1000*float(r['schedule_wall_s'])/int(r['schedule_calls']) for r in rr]
            mm,ss,ms=statistics.mean(m),statistics.mean(s),statistics.mean(pc)
            par=next(r for r in ledger['summaries'] if r['stage']=='confirmation' and r['family']==family and r['arm']==arm)
            for value,field in [(mm,'M_s_mean'),(ss,'schedule_wall_s_mean'),(ms,'mean_schedule_ms_per_call_mean')]:
                assert math.isclose(value,par[field],rel_tol=1e-12,abs_tol=1e-12)
            color=TEAL if arm=='native' else GRAY if arm=='adaptive' else BLUE
            if arm=='reuse':
                ax.axhspan(y-.32,y+.32,color='#E9F0F5',zorder=0);cost.axhspan(y-.32,y+.32,color='#E9F0F5',zorder=0)
            ax.barh(y,ss,height=.43,color='#336787',edgecolor='white',lw=.4,zorder=2)
            ax.barh(y,mm-ss,left=ss,height=.43,color='#DCE3E7',edgecolor='white',lw=.4,zorder=2)
            ax.plot([min(m),max(m)],[y,y],color=color,lw=.8,zorder=3)
            cost.plot([min(pc),max(pc)],[y,y],color=color,lw=.8,zorder=3)
            for j,(v,p) in enumerate(zip(m,pc)):
                jitter=(j-2.5)*.045
                ax.plot(v,y+jitter,'o',ms=2.2,mfc='white',mec=color,mew=.55,zorder=4)
                cost.plot(p,y+jitter,'o',ms=2.2,mfc='white',mec=color,mew=.55,zorder=4)
            cost.text(.5,y+.17,f'{ms:.3f}',ha='center',va='bottom',fontsize=FONT,color=color,transform=cost.get_yaxis_transform())
            record=dict(family=family,arm=arm,repeat=list(range(6)),process_M_s=m,process_schedule_s=s,
                process_ms_per_call=pc,mean_M_s=mm,mean_schedule_s=ss,mean_ms_per_call=ms,displayed_ms_per_call=f'{ms:.3f}')
            records.append(record);local.append(record)
        n=local[0];r=local[2];gap=r['mean_M_s']-n['mean_M_s'];sgap=r['mean_schedule_s']-n['mean_schedule_s'];share=sgap/gap
        par=next(x for x in ledger['same_block_contrasts'] if x['stage']=='confirmation' and x['case']==('confirm_prefix65_interleave' if family=='prefix' else 'confirm_instruction65_interleave') and x['target']=='reuse' and x['baseline']=='native')
        assert math.isclose(share,par['schedule_difference_share_of_mean_M_difference'],rel_tol=1e-12,abs_tol=1e-12)
        derived.append({'family':family,'mean_reuse_on_minus_native_M_s':gap,'mean_reuse_on_minus_native_schedule_s':sgap,
            'recorded_schedule_share_of_gap':share,'displayed_share':f'{100*share:.1f}%',
            'definition':'difference of recorded schedule means / difference of recorded cohort means; accounting, not causal attribution'})
        ax.set_yticks([2,1,0],['Native','Reuse off','Reuse on']);ax.tick_params(axis='y',length=0,pad=4)
        ax.set_xlim(0,25);ax.set_xticks([0,10,20]);ax.set_ylim(-.42,2.75)
        ax.grid(axis='x',color='#E5E9EC',lw=.4,zorder=0);ax.set_xlabel('Cohort time (s)',labelpad=3)
        ax.set_title(title,loc='left',pad=6,fontsize=FONT)
        ax.text(1.63,1.12,f'Gap share: {100*share:.1f}%',ha='right',va='bottom',transform=ax.transAxes,fontsize=10,color=BLUE)
        cost.set_xscale('log');cost.set_xlim(.08,25);cost.set_xticks([.1,1,10],labels=['0.1','1','10'])
        cost.xaxis.set_minor_formatter(NullFormatter());cost.tick_params(which='minor',length=0)
        cost.tick_params(axis='y',left=False,labelleft=False);cost.spines['left'].set_visible(False)
        cost.grid(axis='x',color='#E5E9EC',lw=.4,zorder=0);cost.set_xlabel('ms/call (log)',labelpad=3)
    assert records==parent['displayed']['fig_runtime_ledger_v30']
    fig.legend(handles=[Patch(facecolor='#336787',label='Scheduling'),Patch(facecolor='#DCE3E7',label='Remainder')],
        loc='upper center',bbox_to_anchor=(.54,.995),ncol=2,frameon=False,columnspacing=.8,handlelength=1.2,handletextpad=.35)
    cap=('Fresh-input $4\\times$-quota runtime ledger. Stacked bars show mean cohort time partitioned into '
         'recorded scheduling and arithmetic remainder; open points and segments retain the six repeats and observed ranges. '
         'Right-hand strips show scheduling milliseconds per call, with printed means. Blue bands identify reuse on; '
         'the share labels give the recorded scheduling-interval share of its total-time difference from Native.')
    plain=cap.replace('$','').replace('\\times','x')
    reg=export(fig,'fig_runtime_ledger_v30','fig_runtime_ledger_v22',2,1,plain,cap,
        [d/'runtime_factor4_per_service.csv',d/'RUNTIME_ACCOUNTING.json'],records,
        'Measured stacked wall-time ledger plus observed per-call strips, selected-arm bands and recorded gap-share labels',
        {'formal_services':36,'original_total_time_endpoints':36,'additional_actual_ms_per_call_endpoints':36,
         'per_call_mean_labels':[r['displayed_ms_per_call'] for r in records],'gap_accounting':derived,
         'interpretation':'Arithmetic remainder is not GPU time; scheduling share is recorded accounting, not causal attribution.'})
    return reg,records,derived

def main():
    global W,OUT,FIGS
    parser=argparse.ArgumentParser(description='Rebuild the four current TSMC main experiment figures.')
    parser.add_argument('--manuscript-root',type=Path,default=W)
    W=parser.parse_args().manuscript_root.resolve();OUT=W/'results/figure_build/main';FIGS=W/'results/figure_build/figures'
    setup();OUT.mkdir(parents=True,exist_ok=True);FIGS.mkdir(parents=True,exist_ok=True)
    parent=json.loads((W/'source_data/plot_endpoints/main.json').read_text())
    figs=[];shown={};summary={}
    r,v,s=ownership(parent);figs.append(r);shown[r['new_stem']]=v;summary['ownership_overall']=s
    for k in ['baselines','scale14b']:
        r,v=matrix(k,parent);figs.append(r);shown[r['new_stem']]=v
    r,v,s=runtime(parent);figs.append(r);shown[r['new_stem']]=v;summary['runtime_gap_accounting']=s
    assert shown==parent['displayed']
    assert summary==parent['derived_summaries']
    records={'displayed':shown,'derived_summaries':summary,
             'validation':{'all_source_endpoints_recomputed':True,'all_displayed_records_equal_frozen_endpoints':True,
             'frozen_endpoint_sha256':sha(W/'source_data/plot_endpoints/main.json')}}
    (OUT/'plotrecords.json').write_text(json.dumps(records,indent=2)+'\n')
    for number,row in enumerate(figs,3):row.update(document='main',figure_number=number)
    registry={'release':'1.2.0','figures':figs,'scope':'Four current main figures; no serving, measurement or tuning actions.',
        'plotrecords_sha256':sha(OUT/'plotrecords.json'),'code_sha256':sha(Path(__file__)),
        'validation':'All source ratios/statistics/timer ledgers independently recomputed and matched frozen displayed endpoints.'}
    (OUT/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    print(json.dumps({'status':'PASS','figures':4,'font_family':FONT_FAMILY,'publication_typography':FONT_FAMILY=='Times New Roman','fallback_height_multiplier':HEIGHT_SCALE}))

if __name__=='__main__':main()
