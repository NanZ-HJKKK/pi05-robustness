"""Standalone scientific figures from audited measurements, without synthetic values."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(args.results.read_text());out=args.results.parent
    assert report['status']=='remote_audit_passed_pending_local_archive_acceptance'
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    colors={'Ref':'#777777','B1':'#4c78a8','A1':'#f58518'}
    categories=['Camera Viewpoints','Robot Initial States','Language Instructions','Light Conditions','Background Textures','Sensor Noise','Objects Layout']
    aliases=['Camera','Robot init','Language','Lighting','Background','Noise','Layout']
    lookup={(r['model'],r['kind'],r['group']):r for r in report['quality']}
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True)
    x=np.arange(7);width=.24
    for i,model in enumerate(['Ref','B1','A1']):
        rows=[lookup[(model,'plus',c)] for c in categories]
        axes[0].bar(x+(i-1)*width,[r['success_rate']*100 for r in rows],width,label=model,color=colors[model])
    axes[0].set(xticks=x,xticklabels=aliases,ylim=(0,105),ylabel='Success (%)',title='Heldout LIBERO-Plus variants: 30 episodes per category')
    axes[0].legend(ncol=3,frameon=False);axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    paired=[report['paired_comparisons']['category_A1_minus_B1_'+c] for c in categories]
    delta=np.array([r['treatment_minus_control_pp'] for r in paired]);low=np.array([r['cluster_bootstrap_95_pp'][0] for r in paired]);high=np.array([r['cluster_bootstrap_95_pp'][1] for r in paired])
    axes[1].vlines(x,low,high,color=colors['A1']);axes[1].scatter(x,delta,color=colors['A1'])
    axes[1].hlines(low,x-.06,x+.06,color=colors['A1']);axes[1].hlines(high,x-.06,x+.06,color=colors['A1'])
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set(xticks=x,xticklabels=aliases,ylabel='A1 minus B1 (pp)',title='Paired base-task cluster bootstrap 95% intervals; camera is the registered primary endpoint')
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('One training seed (42); intervals do not measure training-seed variance\nLighting includes one disclosed 1-gray-level RGB exception; strict-subset sensitivity is reported separately.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(out/('robustness.'+ext),dpi=180)
    plt.close(fig)
    state=json.loads((Path(__file__).resolve().parents[1]/'evidence/full_experiment_state.json').read_text())
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for model in ['B1','A1']:
        path=Path(state['models'][model]['run'])/'steps.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        loss=np.array([r['metrics']['loss'] for r in rows]);step=np.array([r['step'] for r in rows])
        smoothed=np.convolve(loss,np.ones(100)/100,mode='valid')
        ax.plot(step,loss,linewidth=.4,alpha=.15,color=colors[model]);ax.plot(step[99:],smoothed,label=model+' / 100-step mean',color=colors[model])
    ax.set(xlabel='Optimizer step',ylabel='Flow-matching loss',title='Training losses use different augmentation conditions; they are not heldout performance')
    ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(frameon=False)
    for ext in ['png','svg']:fig.savefig(out/('training_loss.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for row in report['efficiency']:
        label=f"R{row['replan_steps']}, N{row['num_steps']}";color='#4c78a8' if row['num_steps']==10 else '#f58518'
        for panel,(ax,key) in enumerate(zip(axes,['median_sampling_ms','median_simulation_feedback_hz'])):
            ax.scatter(row[key],row['success_rate']*100,color=color,s=45)
            offsets=[{(1,10):(-45,-34),(5,10):(-90,14),(10,10):(-90,-22),(5,3):(16,-26),(10,3):(16,17)},
                     {(1,10):(-72,-22),(5,10):(-92,-22),(10,10):(14,-22),(5,3):(14,-22),(10,3):(14,16)}]
            offset=offsets[panel][(row['replan_steps'],row['num_steps'])]
            ax.annotate(label,(row[key],row['success_rate']*100),xytext=offset,textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':'#777777','linewidth':.6})
    axes[0].set(xlabel='GPU-synchronized sampling latency (ms)',ylabel='Spatial success (%)',title='Sampling latency and task success',ylim=(92,103))
    axes[1].set(xlabel='Measured simulator feedback frequency (Hz)',ylabel='Spatial success (%)',title='Serial simulator feedback\n(not physical robot Hz)',ylim=(92,103))
    for ax in axes:
        ax.grid(alpha=.2);ax.set_yticks([92,94,96,98,100]);ax.margins(x=.15)
    fig.suptitle('100 Spatial episodes per setting; success axis zoomed to92%. All five registered settings are shown.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(out/('efficiency.'+ext),dpi=180)
    plt.close(fig)
    print('Saved six measured figure artifacts')


if __name__=='__main__':main()
