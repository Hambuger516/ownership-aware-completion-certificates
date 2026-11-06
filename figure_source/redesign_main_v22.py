#!/usr/bin/env python3
"""Four fixed-width scientific figure redesigns; existing measurements only."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import NullFormatter
from pypdf import PdfReader

WIDTH = 3.16697799916978
FONT = 9.0
CELL_FONT = 8.0
COL = {'inc':'#687E88','fixed':'#A37232','adaptive':'#285F8C',
       'native':'#28776F','banker_c0':'#A37232','hard_reserve':'#687E88',
       'schedule':'#336787','remainder':'#DCE3E7'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
gm = lambda values: math.exp(statistics.mean(math.log(x) for x in values))


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def study(root, kind):
    base = root/'source_data/v18_verified'/kind
    data = base/'independent_readout_v1' if (base/'independent_readout_v1').exists() else base
    receipt = json.loads((data/'FINAL_NUMERIC_READOUT.json').read_text())
    assert receipt['status'] == 'COMPLETE_RAW_SOURCE_RESOURCE_MODEL_OUTPUT_WORK_REPLAY_PASS'
    assert receipt['all_formal_outputs_exact'] and receipt['all_controller_reexecution_zero']
    return base, data


def setup():
    for path in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(str(path))
    # Install a licensed Times New Roman font for identical typography; otherwise Matplotlib supplies a fallback.
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm',
        'font.size':FONT,'axes.titlesize':FONT,'axes.labelsize':FONT,
        'xtick.labelsize':FONT,'ytick.labelsize':FONT,'legend.fontsize':FONT,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55,
        'xtick.major.width':.55,'ytick.major.width':.55,'savefig.bbox':None})


def export(fig, root, name, old, rows, cols, height, caption, caption_tex, source, records, points, encoding):
    out = root/'figs';out.mkdir(exist_ok=True)
    fig.canvas.draw()
    visible_text=[a for a in fig.findobj(matplotlib.text.Text) if a.get_text().strip() and a.get_visible()]
    text_sizes = [a.get_fontsize() for a in visible_text]
    assert min(text_sizes) >= 8.0
    assert abs(fig.get_figwidth()-WIDTH) < 1e-12
    canvas=fig.bbox
    for artist in visible_text:
        box=artist.get_window_extent(fig.canvas.get_renderer())
        assert box.x0>=canvas.x0-.5 and box.x1<=canvas.x1+.5 and box.y0>=canvas.y0-.5 and box.y1<=canvas.y1+.5, artist.get_text()
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches=None,pad_inches=0,
                    metadata={'Creator':'Frozen-data scientific figure recipe v22'} if ext=='pdf' else None)
    plt.close(fig)
    pdf=PdfReader(out/f'{name}.pdf');assert len(pdf.pages)==1
    page=pdf.pages[0];width=float(page.mediabox.width);assert abs(width-72*WIDTH)<.001
    fonts=[]
    for value in page['/Resources']['/Font'].values():
        f=value.get_object();assert f['/Subtype']!='/Type3'
        d=f.get('/FontDescriptor')
        if '/DescendantFonts' in f:d=f['/DescendantFonts'][0].get_object().get('/FontDescriptor')
        d=d.get_object();assert any(k in d for k in ['/FontFile','/FontFile2','/FontFile3'])
        fonts.append(str(f['/BaseFont']))
    resources=page['/Resources'];image_count=0
    for value in resources.get('/XObject',{}).get_object().values() if '/XObject'in resources else []:
        if value.get_object().get('/Subtype')=='/Image':image_count+=1
    assert image_count==0
    svg=(out/f'{name}.svg').read_text();assert '<text' in svg and 'Times New Roman' in svg
    return dict(old_stem=old,new_stem=name,outer_rows=rows,outer_columns=cols,rows=rows,cols=cols,span='single',
        width_in=WIDTH,height_in=height,minimum_font_pt=min(text_sizes),preferred_font_pt=FONT,
        caption_plain=caption,caption_tex=caption_tex,encoding=encoding,
        source_sha256={str(p.relative_to(root)):sha(p) for p in source},
        data_paths=[str(p.relative_to(root)) for p in source],
        plot_data_path='source_data/v22_figures/main/plotrows.json',
        raw_displayed_point_count=points,plot_group_count=len(records),
        output_sha256={ext:sha(out/f'{name}.{ext}') for ext in ['pdf','svg','png']},
        figure_sha256={ext:sha(out/f'{name}.{ext}') for ext in ['pdf','svg','png']},
        pdf_fonts_embedded=fonts,pdf_type3_fonts=False,pdf_image_xobjects=image_count,
        svg_live_text_count=svg.count('<text'),fixed_width_no_tight_bbox=True,
        all_visible_text_within_canvas=True,
        visual_signoff=False,records=records)


def ownership(root):
    _,d=study(root,'ownership');raw=read(d/'all_pairs.csv')
    assert len(raw)==36
    services=[r for r in read(d/'all_services.csv') if r['phase']=='timing']
    keyed={(r['case'],int(r['repeat']),r['arm']):r for r in services}
    for pair in raw:
        k=(pair['case'],int(pair['repeat']))
        for metric,field in [('M_s','M_ratio'),('brief_s','brief_ratio')]:
            ratio=float(keyed[k+(pair['target'],)][metric])/float(keyed[k+(pair['baseline'],)][metric])
            assert math.isclose(ratio,float(pair[field]),rel_tol=1e-12,abs_tol=1e-12)
    cases=[f'confirm{i:02}_{p}' for i in range(3) for p in ['front','interleave','back']]
    labels=[f'P{i+1} {p}' for i in range(3) for p in ['F','I','B']]
    height=3.30;fig,axes=plt.subplots(1,2,figsize=(WIDTH,height))
    fig.subplots_adjust(left=.185,right=.97,bottom=.19,top=.83,wspace=.29)
    records=[]
    for col,metric in enumerate(['M_ratio','brief_ratio']):
        ax=axes[col]
        ax.axhspan(2.5,5.5,color='#F2F4F5',zorder=0)
        for y in [1,4,7]:ax.axhspan(y-.43,y+.43,color='#E8F0F6',zorder=0)
        ax.axvline(1,color='#555555',lw=.7,ls=(0,(3,2)),zorder=1)
        for y,case in enumerate(cases):
            rr=sorted([r for r in raw if r['case']==case],key=lambda r:int(r['repeat']))
            assert len(rr)==4 and [int(r['repeat']) for r in rr]==list(range(4))
            assert all(r['target']=='full_c1' and r['baseline']=='full_c0' for r in rr)
            values=[float(r[metric]) for r in rr]
            color=COL['adaptive'] if case.endswith('interleave') else COL['inc']
            ax.plot([min(values),max(values)],[y,y],color=color,lw=1.2,zorder=3)
            ax.scatter(values,[y+(i-1.5)*.105 for i in range(4)],s=8,
                       facecolors='white',edgecolors=color,lw=.65,zorder=4)
            ax.plot(gm(values),y,'D',color=color,ms=3.1,zorder=5)
            records.append(dict(study='ownership',case=case,metric=metric,
                repeat=list(range(4)),raw_ratios=values,GM=gm(values),minimum=min(values),maximum=max(values)))
        ax.set(xscale='log',ylim=(8.6,-.6),yticks=range(9))
        ax.set_yticklabels(labels if col==0 else [])
        ax.tick_params(axis='y',length=0,pad=4)
        if col==0:
            ax.set_xlim(.70,1.06);ax.set_xticks([.7,.8,.9,1],labels=['0.7','0.8','0.9','1'])
        else:
            ax.set_xlim(.045,1.12);ax.set_xticks([.05,.2,1],labels=['0.05','0.2','1'])
        ax.xaxis.set_minor_formatter(NullFormatter());ax.tick_params(which='minor',length=0)
        ax.grid(axis='x',color='#DCE1E4',lw=.4,zorder=0)
        ax.set_title(['(a) Cohort','(b) Brief'][col],pad=7)
    fig.text(.57,.072,'Full C1 / Full C0 (log scale)',ha='center',fontsize=FONT)
    fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=COL['inc'],ms=3,ls='',label='4 paired runs'),
                        Line2D([],[],marker='D',color=COL['adaptive'],ms=3,ls='',label='GM')],
               loc='upper center',bbox_to_anchor=(.53,.992),ncol=2,frameon=False,
               columnspacing=1.0,handletextpad=.35)
    caption=('Ownership ablation on nine fixed input/arrival scenarios. Panels show cohort and brief '
        'completion ratios Full C1/Full C0; F, I and B denote front, interleaved and back arrivals. '
        'Each open point is one matched process block (four per row); diamonds are geometric means '
        'and horizontal segments span observed minima to maxima, not confidence intervals. '
        'Blue row bands identify interleaving; alternating group shading separates the three input pairs. '
        'The dashed line is equality. Both axes are logarithmic; values below one favour Full C1.')
    return export(fig,root,'fig_ownership_repeated_v22','fig_ownership_repeated_v18',1,2,height,
        caption,caption.replace('Full C1/Full C0',r'Full C1/Full C0'),
        [d/n for n in ['all_pairs.csv','all_services.csv','FINAL_NUMERIC_READOUT.json']],records,72,
        'Paired point-and-observed-range forests; group and arrival-category bands')


def ratio_matrix(root, kind):
    base,d=study(root,kind);protocol=json.loads((base/'PROTOCOL.json').read_text())
    cases=protocol['cases'];quotas=protocol['factors'];arms=protocol['arms']
    service=[r for r in read(d/'all_services.csv') if r['phase']=='timing']
    key={(r['case'],int(r['quota_factor']),int(r['repeat']),r['arm']):r for r in service}
    assert len(key)==len(cases)*len(quotas)*4*len(arms)
    source_pairs=read(d/'all_pairs.csv')
    # Every pre-existing pair is rederived, including contrasts not displayed in the compact main matrix.
    for pair in source_pairs:
        k=(pair['case'],int(pair['quota_factor']),int(pair['repeat']))
        for metric,field in [('M_s','M_ratio'),('brief_s','brief_ratio')]:
            ratio=float(key[k+(pair['target'],)][metric])/float(key[k+(pair['baseline'],)][metric])
            assert math.isclose(ratio,float(pair[field]),rel_tol=1e-12,abs_tol=1e-12)
    if kind=='baselines':
        contrast=[('adaptive','banker_c0','Banker C0'),('adaptive','hard_reserve','Hard res.'),('adaptive','native','Native')]
        ratio_name='Adaptive / comparator';old='fig_fair_baselines_v18';name='fig_fair_baselines_v22'
    else:
        contrast=[('inc','native','INC'),('fixed','native','Fixed'),('adaptive','native','Adaptive')]
        ratio_name='Method / Native';old='fig_scale14b_v18';name='fig_scale14b_v22'
    records=[]
    for row,metric in enumerate(['M_s','brief_s']):
        for col,case in enumerate(cases):
            for method,(target,baseline,label) in enumerate(contrast):
                for quota in quotas:
                    values=[float(key[(case,quota,j,target)][metric])/float(key[(case,quota,j,baseline)][metric]) for j in range(4)]
                    records.append(dict(study=kind,case=case,metric=metric,quota=quota,target=target,
                        baseline=baseline,row_label=label,panel_row=row,panel_column=col,method_row=method,
                        raw_ratios=values,repeat=list(range(4)),GM=gm(values),minimum=min(values),maximum=max(values)))
    log_values=[math.log2(v) for r in records for v in r['raw_ratios']]
    lower=min(-1,math.floor(min(log_values)))
    upper=max(1,math.ceil(max(log_values)))
    norm=TwoSlopeNorm(vmin=lower,vcenter=0,vmax=upper)
    cmap=LinearSegmentedColormap.from_list('benefit_cost',['#3D789E','#EEF5F8','#FFFFFF','#F6EDDE','#B78342'])
    height=3.85;fig=plt.figure(figsize=(WIDTH,height));axes=[]
    grid=fig.add_gridspec(2,2,left=.25,right=.985,bottom=.235,top=.875,hspace=.64,wspace=.16)
    point_count=0
    for row in range(2):
        for col,case in enumerate(cases):
            ax=fig.add_subplot(grid[row,col]);axes.append(ax)
            groups=[r for r in records if r['panel_row']==row and r['panel_column']==col]
            values=np.array([[math.log2(next(r['GM'] for r in groups if r['method_row']==method and r['quota']==quota))
                              for quota in quotas] for method in range(3)])
            ax.pcolormesh(np.arange(len(quotas)+1),np.arange(4),values,cmap=cmap,norm=norm,
                          shading='flat',edgecolors='#D4DBDF',linewidth=.45,rasterized=False)
            for r in groups:
                x=quotas.index(r['quota']);y=r['method_row']
                ax.text(x+.5,y+.30,f"{r['GM']:.3f}",ha='center',va='center',fontsize=CELL_FONT,color='#172F3E')
                if kind=='baselines' and row==0 and col==1 and r['baseline']=='hard_reserve' and r['quota']==4:
                    ax.add_patch(Rectangle((x+.015,y+.015),.97,.97,fill=False,
                                           edgecolor='#8E5C24',linewidth=1.05,zorder=6))
                # A common global log-ratio scale is repeated inside each cell.
                xpos=lambda value:x+.12+.76*float(norm(math.log2(value)))
                ax.plot([xpos(1),xpos(1)],[y+.48,y+.94],color='#A5AEB5',ls=(0,(1,1)),lw=.45,zorder=3)
                ax.plot([xpos(r['minimum']),xpos(r['maximum'])],[y+.72,y+.72],color='#243E50',lw=.8,zorder=4)
                for j,value in enumerate(r['raw_ratios']):
                    ax.plot(xpos(value),y+.72+(j-1.5)*.075,'o',ms=1.85,mfc='white',mec='#24465D',mew=.5,zorder=5)
                    point_count+=1
            ax.set_xlim(0,len(quotas));ax.set_ylim(3,0)
            ax.set_xticks(np.arange(len(quotas))+.5,labels=[str(q) for q in quotas])
            ax.set_yticks([.5,1.5,2.5],labels=[x[2] for x in contrast] if col==0 else [])
            ax.tick_params(length=0,pad=3)
            for spine in ax.spines.values():spine.set_visible(False)
            endpoint='cohort' if row==0 else 'brief';family='Prefix' if col==0 else 'Instr.'
            ax.set_title(f'({"abcd"[row*2+col]}) {family} {endpoint}',pad=5,fontsize=FONT)
            if row==1:ax.set_xlabel('Quota / base',labelpad=3)
    # One shared, meaningful-centre log colour/glyph scale for all four panels.
    cax=fig.add_axes([.29,.095,.63,.025])
    sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    cb=fig.colorbar(sm,cax=cax,orientation='horizontal',ticks=[lower,0,upper])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.ax.set_xticklabels([f'1/{2**(-lower):g}','1',f'{2**upper:g}']);cb.ax.tick_params(length=2,pad=1)
    cb.outline.set_linewidth(.4)
    fig.text(.60,.025,'Equality-centred log ratio',ha='center',fontsize=FONT)
    fig.text(.60,.966,ratio_name,ha='center',fontsize=FONT)
    if kind=='baselines':
        caption=('Fair baseline comparison on the 7B model. Rows identify the comparator; each cell is '
            'Adaptive/comparator for one quota and fixed input family, with cohort time above and brief '
            'completion below. The printed value is the geometric mean of four matched process ratios. '
            'Each cell also retains the four open raw points and their observed range using the same '
            'equality-centred log2 glyph scale; its dotted mark is one. Values below and above one '
            'each occupy half of the shared colour/glyph scale. Colour is centred on '
            'equality: blue favours Adaptive and ochre favours the comparator. The outlined instruction '
            '4x-quota cohort cell retains Adaptive/Hard reserve = 1.031. All adverse cells are retained. '
            'These are observed ranges, not confidence intervals; two fixed input families are not independent '
            'workload samples. Full-precision displayed groups and all original method-pair contrasts accompany the figure.')
    else:
        caption=('Parameter-frozen 14B transfer. Rows show INC, Fixed INC-Q and Adaptive INC-Q, each '
            'normalized to matched Native at the two measured quotas. Panels separate fixed input families '
            'and cohort/brief completion. Each printed cell is a geometric mean over four matched process '
            'ratios; four open points and an observed-range segment retain the raw ratios on a shared '
            'equality-centred log2 glyph scale with one marked inside every cell. Values below and '
            'above one each occupy half of the shared colour/glyph scale: blue '
            'is below Native and ochre is above it. Ranges are not confidence intervals. Only the measured '
            'quota levels are shown, with no interpolation, new tuning or projection-reuse result implied.')
    item=export(fig,root,name,old,2,2,height,caption,caption,
        [d/n for n in ['all_pairs.csv','all_services.csv','FINAL_NUMERIC_READOUT.json']]+[base/'PROTOCOL.json'],
        records,point_count,'Annotated geometric-mean vector heatmap plus four-point observed-range glyph per cell')
    item['shared_log2_ratio_limits']=[lower,upper]
    item['shared_ratio_limits']=[2**lower,2**upper]
    item['normalization']='TwoSlopeNorm(log2(ratio)): equality at midpoint; negative and positive spans occupy half each'
    item['small_font_scope']='8pt numeric matrix cell values only; all other text is 9pt'
    item['all_source_pair_contrasts_recomputed']=len(source_pairs)
    item['full_source_pairs']=source_pairs
    item['native_constant_repeats_not_duplicated']=True
    # Preserve the nine quota-by-comparator cross-family GM pairs used by the baseline table.
    if kind=='baselines':
        item['equal_family_baseline_table']=[]
        for quota in quotas:
            for target,baseline,label in contrast:
                item['equal_family_baseline_table'].append(dict(quota=quota,target=target,baseline=baseline,
                    cohort_ratio=gm(r['GM'] for r in records if r['quota']==quota and r['baseline']==baseline and r['metric']=='M_s'),
                    brief_ratio=gm(r['GM'] for r in records if r['quota']==quota and r['baseline']==baseline and r['metric']=='brief_s')))
        assert len(item['equal_family_baseline_table'])==9
    return item


def runtime(root):
    d=root/'experiments/v20_review/runtime_statistics/output';raw=read(d/'runtime_factor4_per_service.csv')
    ledger=json.loads((d/'RUNTIME_ACCOUNTING.json').read_text())
    assert ledger['formal_services']==72 and ledger['all_existing_control_cut_timing_sums_match']
    raw=[r for r in raw if r['stage']=='confirmation'];assert len(raw)==36
    height=3.75;fig=plt.figure(figsize=(WIDTH,height));records=[]
    outer=fig.add_gridspec(2,1,left=.235,right=.975,bottom=.135,top=.865,hspace=.72)
    point_count=0
    for index,(family,title) in enumerate([('prefix','(a) Shared prefix'),('instruction','(b) Instructions')]):
        inner=outer[index].subgridspec(1,2,width_ratios=[1.48,1],wspace=.20)
        ax=fig.add_subplot(inner[0]);cost=fig.add_subplot(inner[1],sharey=ax)
        for y,arm,label in zip([2,1,0],['native','adaptive','reuse'],['Native','Reuse off','Reuse on']):
            rr=sorted([r for r in raw if r['family']==family and r['arm']==arm],key=lambda r:int(r['repeat']))
            assert len(rr)==6 and [int(r['repeat']) for r in rr]==list(range(6))
            m=[float(r['M_s']) for r in rr];s=[float(r['schedule_wall_s']) for r in rr]
            percall=[1000*float(r['schedule_wall_s'])/int(r['schedule_calls']) for r in rr]
            mm,ss,ms=statistics.mean(m),statistics.mean(s),statistics.mean(percall)
            parent=next(r for r in ledger['summaries'] if r['stage']=='confirmation' and r['family']==family and r['arm']==arm)
            assert math.isclose(ms,parent['mean_schedule_ms_per_call_mean'],rel_tol=1e-12)
            color=COL['native'] if arm=='native' else COL['inc'] if arm=='adaptive' else COL['adaptive']
            ax.barh(y,ss,height=.47,color=COL['schedule'],edgecolor='white',lw=.4,zorder=2)
            ax.barh(y,mm-ss,left=ss,height=.47,color=COL['remainder'],edgecolor='white',lw=.4,zorder=2)
            ax.plot([min(m),max(m)],[y,y],color=color,lw=.8,zorder=3)
            cost.plot([min(percall),max(percall)],[y,y],color=color,lw=.8,zorder=3)
            for j,(value,pc) in enumerate(zip(m,percall)):
                jitter=(j-2.5)*.046
                ax.plot(value,y+jitter,'o',ms=2,mfc='white',mec=color,mew=.55,zorder=4)
                cost.plot(pc,y+jitter,'o',ms=2,mfc='white',mec=color,mew=.55,zorder=4)
                point_count+=1
            cost.text(.5,y+.20,f'{ms:.3f}',ha='center',va='bottom',fontsize=FONT,
                      color=color,transform=cost.get_yaxis_transform())
            records.append(dict(family=family,arm=arm,repeat=list(range(6)),process_M_s=m,process_schedule_s=s,
                process_ms_per_call=percall,mean_M_s=mm,mean_schedule_s=ss,mean_ms_per_call=ms,
                displayed_ms_per_call=f'{ms:.3f}'))
        ax.set_yticks([2,1,0],['Native','Reuse off','Reuse on']);ax.tick_params(axis='y',length=0,pad=4)
        ax.set_xlim(0,25);ax.set_xticks([0,10,20]);ax.set_ylim(-.42,2.65)
        ax.grid(axis='x',color='#E5E9EC',lw=.4,zorder=0);ax.set_xlabel('Cohort time (s)',labelpad=3)
        ax.set_title(title,loc='left',pad=6,fontsize=FONT)
        cost.set_xscale('log');cost.set_xlim(.08,25);cost.set_xticks([.1,1,10],labels=['0.1','1','10'])
        cost.xaxis.set_minor_formatter(NullFormatter());cost.tick_params(which='minor',length=0)
        cost.tick_params(axis='y',left=False,labelleft=False);cost.spines['left'].set_visible(False)
        cost.grid(axis='x',color='#E5E9EC',lw=.4,zorder=0);cost.set_xlabel('ms/call (log)',labelpad=3)
    fig.legend(handles=[Patch(facecolor=COL['schedule'],label='Scheduling'),
                        Patch(facecolor=COL['remainder'],label='Remainder')],
               loc='upper center',bbox_to_anchor=(.54,.992),ncol=2,frameon=False,
               columnspacing=.8,handlelength=1.2,handletextpad=.35)
    caption=('Fresh-input 4x-quota scheduling ledger, with prefix and instruction inputs shown in stacked '
        'panels. Left: mean cohort wall time over six matched balanced process blocks, split into the '
        'recorded scheduling interval and the arithmetic remainder; all 36 original total-time endpoints '
        'are retained as open points. Right: scheduling milliseconds per invocation for the same runs, '
        'with each arm mean printed above its six raw points on a logarithmic axis. Segments span observed '
        'ranges, not confidence intervals. Reuse off/on share the same policy. The remainder is not '
        'measured GPU time and recorded scheduling is not an irreducible certificate lower bound.')
    item=export(fig,root,'fig_runtime_ledger_v22','fig_runtime_ledger_v21',2,1,height,
        caption,caption.replace('4x-quota',r'4$\times$-quota'),
        [d/'runtime_factor4_per_service.csv',d/'RUNTIME_ACCOUNTING.json'],records,point_count*2,
        'Stacked measured wall-time ledger and six-run ms/call observed-point strips; two stacked panels')
    item['original_total_time_endpoints']=point_count
    item['additional_actual_ms_per_call_endpoints']=point_count
    item['per_call_mean_labels']=[r['displayed_ms_per_call'] for r in records]
    return item


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manuscript-root',type=Path,required=True)
    args=p.parse_args();root=args.manuscript_root.resolve();setup()
    items=[ownership(root),ratio_matrix(root,'baselines'),ratio_matrix(root,'scale14b'),runtime(root)]
    out=root/'source_data/v22_figures/main';out.mkdir(parents=True,exist_ok=True)
    rows={r['new_stem']:r.pop('records') for r in items}
    # The full original pair table is transparent companion data, not filtered or selected.
    companions={r['new_stem']:r.pop('full_source_pairs') for r in items if 'full_source_pairs'in r}
    (out/'plotrows.json').write_text(json.dumps(dict(displayed=rows,complete_original_pair_tables=companions),indent=2)+'\n')
    flat=[]
    for figure,groups in rows.items():
        for group in groups:
            if 'raw_ratios'in group:
                for repeat,value in enumerate(group['raw_ratios']):
                    flat.append(dict(figure=figure,case=group.get('case',''),family=group.get('family',''),
                        metric=group['metric'],quota=group.get('quota',1),target=group.get('target','full_c1'),
                        baseline=group.get('baseline','full_c0'),repeat=repeat,value=value,
                        center=group['GM'],minimum=group['minimum'],maximum=group['maximum']))
            else:
                for metric,field,center in [('M_s','process_M_s','mean_M_s'),('schedule_s','process_schedule_s','mean_schedule_s'),
                                             ('ms_per_call','process_ms_per_call','mean_ms_per_call')]:
                    for repeat,value in enumerate(group[field]):
                        flat.append(dict(figure=figure,case='fresh_confirmation',family=group['family'],metric=metric,
                            quota=4,target=group['arm'],baseline='',repeat=repeat,value=value,center=group[center],
                            minimum=min(group[field]),maximum=max(group[field])))
    with (out/'plotrows.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    companion_dir=out/'plotrows';companion_dir.mkdir(exist_ok=True)
    for kind in ['baselines','scale14b']:
        _,data=study(root,kind)
        item=next(x for x in items if x['new_stem']==('fig_fair_baselines_v22' if kind=='baselines' else 'fig_scale14b_v22'))
        item['transparent_companion_files']={}
        for name in ['all_services.csv','all_pairs.csv']:
            path=companion_dir/f'{kind}_{name}';path.write_bytes((data/name).read_bytes())
            assert sha(path)==sha(data/name)
            item['transparent_companion_files'][str(path.relative_to(root))]=sha(path)
    registry=dict(version='v22',single_column_width_inches=WIDTH,figures=items,
        editability='Scientific source and live-text SVG plus vector PDF; not a native PowerPoint chart.',
        source_and_algorithm_changes=False,new_GPU_runs=False)
    (out/'LAYOUT_AND_CAPTIONS.json').write_text(json.dumps(registry,indent=2)+'\n')
    qa=dict(status='FOUR_MAIN_FIGURES_FIXED_WIDTH_NUMERICAL_VECTOR_CHECKS_PASS_VISUAL_PENDING',
        recipe_sha256=sha(Path(__file__)),minimum_font_pt=min(x['minimum_font_pt'] for x in items),
        all_widths_exact=True,all_source_pairs_recomputed=True,no_source_measurement_changed=True,
        no_invented_CI_KDE_or_interpolation=True,visual_signoff=False,
        figures={x['new_stem']:{k:x[k] for k in ['width_in','height_in','raw_displayed_point_count',
             'pdf_fonts_embedded','pdf_type3_fonts','pdf_image_xobjects','svg_live_text_count','output_sha256']} for x in items})
    (out/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':main()
