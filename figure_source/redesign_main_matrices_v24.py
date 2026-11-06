#!/usr/bin/env python3
"""Rebuild mean-only matrices and all paired/range CSVs from public analysis-ready rows."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from pypdf import PdfReader

WIDTH=3.16697799916978
HEIGHT=3.25
FONT=9.0
LOG2_LIMITS=[-6,0,2]
COLOURS=['#3D789E','#EEF5F8','#FFFFFF','#F6EDDE','#B78342']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gm=lambda values:math.exp(statistics.mean(math.log(v)for v in values))


def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def study(root,kind):
    base=root/'source_data/v18_verified'/kind
    data=base/'independent_readout_v1' if (base/'independent_readout_v1').exists() else base
    j=json.loads((data/'FINAL_NUMERIC_READOUT.json').read_text())
    assert j['status']=='COMPLETE_RAW_SOURCE_RESOURCE_MODEL_OUTPUT_WORK_REPLAY_PASS'
    assert j['all_formal_outputs_exact'] and j['all_controller_reexecution_zero']
    return base,data


def setup():
    for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(str(p))
    assert 'Times New Roman'in {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'cm','font.size':FONT,
        'axes.titlesize':FONT,'axes.labelsize':FONT,'xtick.labelsize':FONT,'ytick.labelsize':FONT,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.55})


def compute(root,kind,parent):
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
    parent_stem='fig_fair_baselines_v23'if kind=='baselines'else'fig_scale14b_v23'
    assert groups==parent['displayed'][parent_stem], 'The independently recomputed process/statistic records differ from v23'
    sources=[data/name for name in ['all_services.csv','all_pairs.csv','FINAL_NUMERIC_READOUT.json']]+[base/'PROTOCOL.json']
    return groups,quotas,contrasts,sources,len(source_pairs)


def draw(root,kind,groups,quotas,contrasts,sources):
    name='fig_fair_baselines_v24'if kind=='baselines'else'fig_scale14b_v24'
    old='fig_fair_baselines_v23'if kind=='baselines'else'fig_scale14b_v23'
    ratio_name='Adaptive / comparator'if kind=='baselines'else'Method / Native'
    norm=TwoSlopeNorm(vmin=-6,vcenter=0,vmax=2)
    cmap=LinearSegmentedColormap.from_list('benefit_cost_v24',COLOURS)
    fig=plt.figure(figsize=(WIDTH,HEIGHT))
    grid=fig.add_gridspec(2,2,left=.25,right=.985,bottom=.245,top=.85,hspace=.67,wspace=.16)
    for row in range(2):
        for col in range(2):
            ax=fig.add_subplot(grid[row,col]);rs=[r for r in groups if r['panel_row']==row and r['panel_column']==col]
            values=np.array([[math.log2(next(r['GM']for r in rs if r['method_row']==m and r['quota']==q))for q in quotas]for m in range(3)])
            assert np.all(values>=-6) and np.all(values<=2)
            ax.pcolormesh(np.arange(len(quotas)+1),np.arange(4),values,cmap=cmap,norm=norm,
                          shading='flat',edgecolors='#D4DBDF',linewidth=.45,rasterized=False)
            for r in rs:
                x=quotas.index(r['quota']);y=r['method_row']
                rgb=cmap(norm(math.log2(r['GM'])))[:3]
                linear=[c/12.92 if c<=.04045 else((c+.055)/1.055)**2.4 for c in rgb]
                luminance=sum(c*w for c,w in zip(linear,[.2126,.7152,.0722]))
                ink='white'if luminance<.23 else'#172F3E'
                ax.text(x+.5,y+.5,f"{r['GM']:.3f}",ha='center',va='center',fontsize=FONT,color=ink)
                if kind=='baselines'and row==0 and col==1 and r['baseline']=='hard_reserve'and r['quota']==4:
                    ax.add_patch(Rectangle((x+.015,y+.015),.97,.97,fill=False,edgecolor='#8E5C24',linewidth=1.05,zorder=6))
            # Main matrix cells deliberately have zero raw-point/range/reference glyphs.
            assert not ax.lines and not any(isinstance(c,matplotlib.collections.PathCollection)for c in ax.collections)
            ax.set_xlim(0,len(quotas));ax.set_ylim(3,0)
            ax.set_xticks(np.arange(len(quotas))+.5,labels=[str(q)for q in quotas])
            ax.set_yticks([.5,1.5,2.5],labels=[r[2]for r in contrasts]if col==0 else[])
            ax.tick_params(length=0,pad=3)
            for spine in ax.spines.values():spine.set_visible(False)
            family='Prefix'if col==0 else'Instr.';endpoint='cohort'if row==0 else'brief'
            ax.set_title(f'({"abcd"[row*2+col]}) {family} {endpoint}',pad=5)
            if row==1:ax.set_xlabel('Quota / base',labelpad=3)
    cax=fig.add_axes([.29,.095,.63,.025]);sm=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    cb=fig.colorbar(sm,cax=cax,orientation='horizontal',ticks=LOG2_LIMITS)
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face')
    cb.ax.set_xticklabels(['1/64','1','4']);cb.ax.tick_params(length=2,pad=1);cb.outline.set_linewidth(.4)
    fig.text(.60,.025,'Equality-centred log ratio',ha='center')
    fig.text(.60,.966,ratio_name,ha='center')
    fig.canvas.draw();canvas=fig.bbox;renderer=fig.canvas.get_renderer()
    texts=[a for a in fig.findobj(matplotlib.text.Text)if a.get_visible()and a.get_text().strip()]
    assert min(a.get_fontsize()for a in texts)>=8
    for a in texts:
        b=a.get_window_extent(renderer)
        assert b.x0>=canvas.x0-.5 and b.y0>=canvas.y0-.5 and b.x1<=canvas.x1+.5 and b.y1<=canvas.y1+.5,a.get_text()
    out=root/'figs'
    for ext in ['pdf','svg','png']:fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches=None,pad_inches=0)
    plt.close(fig)
    page=PdfReader(out/f'{name}.pdf').pages[0]
    assert math.isclose(float(page.mediabox.width)/72,WIDTH,abs_tol=1e-6)
    fonts=[]
    for value in page['/Resources']['/Font'].values():
        f=value.get_object();assert f['/Subtype']!='/Type3';d=f.get('/FontDescriptor')
        if'/DescendantFonts'in f:d=f['/DescendantFonts'][0].get_object().get('/FontDescriptor')
        d=d.get_object();assert any(n in d for n in ['/FontFile','/FontFile2','/FontFile3']);fonts.append(str(f['/BaseFont']))
    xobjects=page['/Resources'].get('/XObject',{})
    assert all(v.get_object().get('/Subtype')!='/Image'for v in xobjects.get_object().values())if xobjects else True
    svg=(out/f'{name}.svg').read_text();assert'<text'in svg and'Times New Roman'in svg
    if kind=='baselines':
        label='tab:matrix-baseline-ranges-v24'
        plain=('7B Adaptive/comparator ratios: cells show geometric means of four matched process pairs. '
            'Blue/ochre indicate below/above one. Both matrices share the equality-centred log2 colour scale '
            '(1/64, 1, 4), with half the colour range on each side. The outlined instruction 4x cohort ratio is 1.031. '
            'Observed repeat ranges and paired values accompany the supplementary baseline-range table.')
        tex=(r'7B Adaptive/comparator ratios: cells show geometric means of four matched process pairs. '
            r'Blue/ochre indicate below/above one. Both matrices share the equality-centred $\log_2$ colour scale '
            r'$(1/64,1,4)$, with half the colour range on each side. The outlined instruction $4\times$ cohort ratio is 1.031. '
            r'Observed repeat ranges: Supplementary Table~\ref{tab:matrix-baseline-ranges-v24}.')
    else:
        label='tab:matrix-14b-ranges-v24'
        plain=('14B method/Native geometric-mean ratios over four matched process pairs per cell. '
            'The colour direction and shared equality-centred log2 scale (1/64, 1, 4) match the baseline matrix. '
            'Observed repeat ranges and paired values accompany the supplementary 14B-range table.')
        tex=(r'14B method/Native geometric-mean ratios over four matched process pairs per cell. '
            r'The colour direction and shared equality-centred $\log_2$ scale $(1/64,1,4)$ match Fig.~\ref{fig:quota-main}. '
            r'Observed repeat ranges: Supplementary Table~\ref{tab:matrix-14b-ranges-v24}.')
    return dict(old_stem=old,new_stem=name,rows=2,cols=2,span='single',width_in=WIDTH,height_in=HEIGHT,
        minimum_font_pt=min(a.get_fontsize()for a in texts),caption_plain=plain,caption_tex=tex,SI_range_label=label,
        shared_log2_ratio_limits=LOG2_LIMITS,shared_ratio_limits=[1/64,1,4],cmap_colours=COLOURS,
        ink_luminance_threshold=.23,ink_colours=['white','#172F3E'],
        normalization='TwoSlopeNorm(log2(ratio),vmin=-6,vcenter=0,vmax=2); each side of equality occupies half of colour range',
        main_raw_points=0,main_range_segments=0,main_dotted_cell_markers=0,all_cell_values_centred=True,
        main_cell_gm_font_pt=FONT,cell_glyphs_removed=True,all_original_process_values_retained_in_companion_data=True,
        source_sha256={str(p.relative_to(root)):sha(p)for p in sources},
        plot_data_path='source_data/v24_checks/matrices/endpoint_groups.json',
        figure_sha256={ext:sha(out/f'{name}.{ext}')for ext in ['pdf','svg','png']},
        fonts_embedded=fonts,Type3_fonts=False,whole_plot_raster=False,svg_live_text_count=svg.count('<text'),
        all_visible_text_inside_canvas=True,visual_signoff=False)


def write_csv(path,rows):
    with path.open('w',newline='')as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def si_table(kind,groups):
    baseline=kind=='baselines';label='tab:matrix-baseline-ranges-v24'if baseline else'tab:matrix-14b-ranges-v24'
    caption=(r'Observed process variation for the 7B baseline matrix. Rows compare Adaptive with the named comparator. '
             if baseline else r'Observed process variation for the parameter-frozen 14B matrix. Rows compare the named method with Native. ')
    caption+=(r'Each entry is the geometric mean [minimum, maximum] of four matched same-block process ratios; '
              r'$M$ is cohort time and $B$ is brief completion time. The four original paired ratios per endpoint are retained in the companion CSV.')
    lines=[r'\begin{table*}[!t]',r'\centering\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.04}',
        r'\caption{'+caption+r'}\label{'+label+'}',r'\begin{tabular}{@{}lllrr@{}}\toprule',
        'Input & Quota & '+('Comparator'if baseline else'Method / Native')+r' & $M$: GM [min, max] & $B$: GM [min, max]\\\midrule']
    combined=[]
    identities=[]
    for r in groups:
        key=(r['case'],r['quota'],r['target'],r['baseline'])
        if key not in identities:identities.append(key)
    family_order={'fresh00_interleave':0,'alpaca_new00_interleave':1}
    comparator_order={'banker_c0':0,'hard_reserve':1,'native':2}
    method_order={'inc':0,'fixed':1,'adaptive':2}
    identities.sort(key=lambda k:(family_order[k[0]],k[1],comparator_order[k[3]]if baseline else method_order[k[2]]))
    labels={'banker_c0':'Banker C0','hard_reserve':'Hard reserve','native':'Native',
            'inc':'INC','fixed':'Fixed INC-Q','adaptive':'Adaptive INC-Q'}
    for case,quota,target,reference in identities:
        metrics={r['metric']:r for r in groups if(r['case'],r['quota'],r['target'],r['baseline'])==(case,quota,target,reference)}
        assert set(metrics)=={'M_s','brief_s'}
        family='Prefix'if case=='fresh00_interleave'else'Instructions'
        field=lambda r:f"{r['GM']:.4f} [{r['minimum']:.4f}, {r['maximum']:.4f}]"
        text_label=labels[reference if baseline else target]
        lines.append(family+f' & {quota}'+r'$\times$ & '+text_label+' & '+field(metrics['M_s'])+' & '+field(metrics['brief_s'])+r' \\')
        row=dict(study=kind,case=case,family=family,quota=quota,target=target,baseline=reference,n_matched_process_pairs=4)
        for metric,short in [('M_s','M'),('brief_s','B')]:
            r=metrics[metric];row.update({short+'_GM':r['GM'],short+'_minimum':r['minimum'],short+'_maximum':r['maximum']})
            row.update({short+f'_ratio_r{j}':v for j,v in enumerate(r['raw_ratios'])})
        combined.append(row)
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table*}'])
    assert len(combined)==(18 if baseline else 12)
    return '\n'.join(lines),combined


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manuscript-root',type=Path,required=True)
    root=p.parse_args().manuscript_root.resolve();out=root/'source_data/v24_checks/matrices';out.mkdir(parents=True,exist_ok=True)
    parent_path=root/'source_data/v23_concision/matrix_redraw/plotrows.json';parent=json.loads(parent_path.read_text())
    setup();registry=[];allgroups=[];tables=[];longrows=[];recomputed=0
    for kind in ['baselines','scale14b']:
        groups,quotas,contrasts,sources,count=compute(root,kind,parent);recomputed+=count
        registry.append(draw(root,kind,groups,quotas,contrasts,sources));allgroups.extend(groups)
        latex,combined=si_table(kind,groups);tables.append(latex)
        write_csv(out/f'{kind}_GM_observed_ranges.csv',combined)
        for r in groups:
            for repeat,value in enumerate(r['raw_ratios']):
                longrows.append(dict(study=kind,case=r['case'],quota=r['quota'],target=r['target'],baseline=r['baseline'],
                    metric=r['metric'],repeat=repeat,paired_ratio=value,GM=r['GM'],minimum=r['minimum'],maximum=r['maximum']))
    assert len(allgroups)==60 and len(longrows)==240
    write_csv(out/'all_four_paired_ratios.csv',longrows)
    (out/'endpoint_groups.json').write_text(json.dumps(allgroups,indent=2)+'\n')
    # The public export retains the complete table data in CSV, without manuscript TeX.
    for item in registry:item['observed_range_csv']=f"source_data/v24_checks/matrices/{'baselines' if item['new_stem']=='fig_fair_baselines_v24' else 'scale14b'}_GM_observed_ranges.csv"
    (out/'REGISTRY.json').write_text(json.dumps(dict(version='v24',figures=registry),indent=2)+'\n')
    qa=dict(status='V24_MEAN_ONLY_MATRICES_NUMERICAL_VECTOR_FONT_GEOMETRY_PASS_VISUAL_PENDING',pass_=True,
        recipe_sha256=sha(Path(__file__)),source_v23_plotrows_sha256=sha(parent_path),
        endpoint_groups=60,original_paired_endpoint_values=240,all_v23_plot_data_exactly_equal=True,
        all_original_pair_contrasts_independently_recomputed=recomputed,
        all_source_pair_endpoint_ratios_independently_recomputed=recomputed*2,
        shared_TwoSlopeNorm_log2=LOG2_LIMITS,cmap_colours=COLOURS,source_data_changed=False,algorithm_changed=False,new_GPU_services=0,
        main_points_or_range_or_dotted_cell_glyphs=0,all_fonts_at_least_8pt=True,all_text_9pt=True,
        SI_baseline_rows=18,SI_14B_rows=12,SI_table_data_in_public_CSV=True,SI_range_labels=[r['SI_range_label']for r in registry],
        data_sha256={p.name:sha(p)for p in sorted(out.glob('*.csv'))+[out/'endpoint_groups.json']},
        figure_sha256={r['new_stem']:r['figure_sha256']for r in registry},visual_signoff=False)
    (out/'QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    (out/'README.txt').write_text('V24 mean-only matrices\n\nCells show geometric means of four matched process pairs at fixed 9 pt. Both 2x2 single-column figures share the equality-centred log2 ratio scale (1/64, 1, 4). Blue/ochre indicate below/above one. The adverse 1.031 baseline cell remains outlined.\n\nendpoint_groups.json contains 60 endpoint groups. all_four_paired_ratios.csv retains all 240 original ratios. baselines_GM_observed_ranges.csv and scale14b_GM_observed_ranges.csv contain 18 and 12 rows respectively, with both endpoints, geometric means, observed minima/maxima and all four paired ratios. These observed ranges are not confidence intervals. Every group is independently recomputed from the public frozen measurements and equals the v23 plotted record.\n\nReproduce: python figure_source/redesign_main_matrices_v24.py --manuscript-root .\nThe public recipe emits figures and complete table data as CSV; manuscript TeX is excluded. It uses matplotlib, numpy, pypdf and installed Times New Roman fonts.\n')
    print(json.dumps(qa,indent=2))


if __name__=='__main__':main()
