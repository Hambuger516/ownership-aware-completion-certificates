#!/usr/bin/env python3
"""Render the two main ratio matrices with non-touching repeat markers.

Scientific observations, geometric means, scales and physical geometry are
unchanged from v22. Only the repeat-marker typography and captions are revised.
"""
from pathlib import Path
import argparse, json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from redesign_main_v22 import CELL_FONT, FONT, WIDTH, export, gm, read, setup, study

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
        ratio_name='Adaptive / comparator';old='fig_fair_baselines_v22';name='fig_fair_baselines_v23'
    else:
        contrast=[('inc','native','INC'),('fixed','native','Fixed'),('adaptive','native','Adaptive')]
        ratio_name='Method / Native';old='fig_scale14b_v22';name='fig_scale14b_v23'
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
                rgb=cmap(norm(math.log2(r['GM'])))[:3]
                linear=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in rgb]
                luminance=sum(c*w for c,w in zip(linear,[.2126,.7152,.0722]))
                ink='white' if luminance<.23 else '#172F3E'
                ax.text(x+.5,y+.30,f"{r['GM']:.3f}",ha='center',va='center',fontsize=CELL_FONT,color=ink)
                if kind=='baselines' and row==0 and col==1 and r['baseline']=='hard_reserve' and r['quota']==4:
                    ax.add_patch(Rectangle((x+.015,y+.015),.97,.97,fill=False,
                                           edgecolor='#8E5C24',linewidth=1.05,zorder=6))
                # A common global log-ratio scale is repeated inside each cell.
                xpos=lambda value:x+.12+.76*float(norm(math.log2(value)))
                ax.plot([xpos(1),xpos(1)],[y+.48,y+.94],color='#A5AEB5',ls=(0,(1,1)),lw=.45,zorder=3)
                ax.plot([xpos(r['minimum']),xpos(r['maximum'])],[y+.72,y+.72],color=ink,lw=.8,zorder=4)
                for j,value in enumerate(r['raw_ratios']):
                    ax.plot(xpos(value),y+.72+(j-1.5)*.11,'o',ms=1.25,mfc=ink,mec=ink,mew=0,zorder=5)
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
        caption=('Fair baseline comparison on the 7B model. Each cell shows Adaptive/comparator: '
            'the printed geometric mean, four paired process points and their observed range. '
            'Panels separate input family and cohort/brief time. Blue denotes ratios below one '
            'and ochre above one; each side occupies half of the shared log2 colour/point scale. '
            'The dotted cell marker is equality. The outlined cell highlights the 1.031 cohort ratio '
            'against hard reservation at 4x quota.')
    else:
        caption=('Parameter-frozen 14B transfer. Rows compare INC, Fixed and Adaptive with matched Native '
            'at the two measured quotas; panels separate input family and cohort/brief time. '
            'Each cell gives the geometric mean, four paired process points and their observed range. '
            'Blue denotes ratios below one and ochre above one; each side occupies half of the '
            'shared log2 colour/point scale. The dotted cell marker is equality.')
    item=export(fig,root,name,old,2,2,height,caption,caption,
        [d/n for n in ['all_pairs.csv','all_services.csv','FINAL_NUMERIC_READOUT.json']]+[base/'PROTOCOL.json'],
        records,point_count,'Annotated geometric-mean vector heatmap with four non-touching filled process points and observed range')
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

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript-root',type=Path,required=True)
    root=parser.parse_args().manuscript_root.resolve()
    setup()
    rows=[ratio_matrix(root,kind)for kind in ['baselines','scale14b']]
    data={r['new_stem']:r.pop('records')for r in rows}
    companions={r['new_stem']:r.pop('full_source_pairs')for r in rows}
    prior=json.loads((root/'source_data/v22_figures/main/plotrows.json').read_text())['displayed']
    for stem,points in data.items():
        assert points==prior[stem.replace('_v23','_v22')], 'Scientific matrix values changed'
    out=root/'source_data/v23_concision/matrix_redraw'
    out.mkdir(parents=True,exist_ok=True)
    (out/'plotrows.json').write_text(json.dumps(dict(displayed=data,complete_original_pair_tables=companions),indent=2)+'\n')
    for r in rows:
        r['plot_data_path']='source_data/v23_concision/matrix_redraw/plotrows.json'
        r['repeat_marker_diameter_pt']=1.25
        r['repeat_marker_vertical_spacing_cell']=.11
        r['no_observations_or_statistics_changed']=True
    (out/'REGISTRY.json').write_text(json.dumps(dict(version='v23',figures=rows),indent=2)+'\n')
    (out/'QA.json').write_text(json.dumps(dict(status='TWO_MATRICES_EXACT_PARENT_VALUES_VECTOR_FONT_GEOMETRY_PASS_VISUAL_PENDING',visual_signoff=False,all_parent_plotted_values_identical=True,repeat_point_diameter_pt=1.25,all_source_pair_contrasts_recomputed=True,new_GPU_services=0),indent=2)+'\n')
    print('Two v23 matrices generated; all original numerical records unchanged')

if __name__=='__main__':main()
