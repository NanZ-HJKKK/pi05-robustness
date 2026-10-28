"""Audit complete scored runs and compute registered paired comparisons."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pi05_robustness.statistics import summarize,paired_comparison
from evaluate_v3 import plan_tasks,request_seed


def load_job(job):
    out=Path(job['out']);result=json.loads((out/'result.json').read_text())
    assert result['status']=='complete' and result['owned_processes_stopped']
    registration_path=ROOT/'configs/inference_reproducibility_registration.json'
    registration=json.loads(registration_path.read_text())
    assert result['inference_reproducibility_registration_sha256']==hashlib.sha256(registration_path.read_bytes()).hexdigest()
    directories=sorted((out/'episodes').glob('*/run.json')) if not job['serial'] else [out/'episodes/run.json']
    rows=[]
    for metadata_path in directories:
        metadata=json.loads(metadata_path.read_text());settings=metadata['settings']
        server=metadata['server']
        assert server['model_id']==job['model'] and server['num_steps']==job['num_steps'] and server['latency_synchronized']
        assert server['config']==('pi05_libero' if job['model']=='Ref' else 'pi05_libero_lora_ft')
        cache=registration['caches'][server['config']+'_N'+str(job['num_steps'])]
        assert server['inference_autotune_cache_sha256']==cache['sha256']==result['inference_autotune_cache_sha256']
        assert '--xla_gpu_require_complete_aot_autotune_results=true' in server['xla_flags']
        assert metadata['evaluator_filename']=='evaluate_v3.py'
        assert settings['kind']==job['kind'] and settings['model_id']==job['model']
        assert settings['condition']=='normal' and settings['replan_steps']==job['replan_steps'] and settings['num_steps']==job['num_steps']
        if job['kind']=='plus':assert settings['plus_split']=='heldout'
        attempts=[json.loads(x) for x in metadata_path.with_name('episodes.jsonl').read_text().splitlines()]
        assert len(attempts)==metadata['expected_episodes'] and all(r['status']=='complete' for r in attempts)
        for row in attempts:
            n=row['environment_steps']
            assert len(row['actions'])==len(row['rewards'])==len(row['done_signals'])==n and n>0
            assert all(len(action)==7 and all(math.isfinite(value) for value in action) for action in row['actions'])
            assert all(math.isfinite(x) for x in row['rewards'])
            assert bool(any(row['done_signals']))==row['success']
            if row['success']:assert row['done_signals'][-1] and sum(row['done_signals'])==1
            assert [t['environment_step'] for t in row['inference']]==list(range(0,n,job['replan_steps']))
            for timing in row['inference']:
                assert timing['request_seed_uint32']==request_seed(row['flow_noise_identity'],timing['environment_step'],row['seed'])
                assert all(math.isfinite(x) for x in timing['observed_proprio']) and len(timing['observed_proprio'])==8
            assert Path(row['video']).is_file()
        rows.extend(attempts)
    expected=plan_tasks(job['kind'],'libero_spatial' if job['serial'] else 'all',10 if job['kind']=='clean' else 1,'heldout')
    fields=lambda r:(r['suite'],r['task_name'],r['init_state_index'])
    assert len(rows)==len(expected) and {fields(r) for r in rows}=={fields(r) for r in expected}
    assert result['episodes_completed']==len(rows) and result['successes']==sum(r['success'] for r in rows)
    requests=[t for r in rows for t in r['inference']]
    assert sorted(t['request_index'] for t in requests)==list(range(len(requests)))
    assert all(t['warmup']==(t['request_index']<5) for t in requests)
    assert len({t['server_run_id'] for t in requests})==1
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'runs/final_analysis');args=p.parse_args()
    state=json.loads((ROOT/'evidence/full_experiment_state.json').read_text())
    assert state['status']=='all_jobs_complete_pending_independent_acceptance'
    assert len(state['evaluations'])==11 and state['matched_training_initialization_and_first10_inputs']
    args.out.mkdir(parents=True,exist_ok=True)
    exception_path=ROOT/'evidence/render_rounding_exception.json'
    exception=json.loads(exception_path.read_text())
    amendment=json.loads((ROOT/'configs/analysis_integrity_amendment.json').read_text())
    assert hashlib.sha256(exception_path.read_bytes()).hexdigest()==amendment['exception_proof_sha256']
    assert exception['original_records_unchanged'] and not exception['original_strict_pixel_gate_passed']
    quality={};efficiency={};coverage=[];table=[]
    for job in state['evaluations']:
        rows=load_job(job);coverage.append({**job,'episodes':len(rows),'successes':sum(r['success'] for r in rows)})
        if job['serial']:efficiency[(job['replan_steps'],job['num_steps'])]=rows
        else:
            quality[(job['model'],job['kind'])]=rows
            table.append({'model':job['model'],'kind':job['kind'],'group':'overall',**summarize(rows)})
            by_group=defaultdict(list)
            for row in rows:by_group[row['suite'] if job['kind']=='clean' else row['category']].append(row)
            for group,group_rows in sorted(by_group.items()):table.append({'model':job['model'],'kind':job['kind'],'group':group,**summarize(group_rows)})
    comparisons={}
    for kind in ['clean','plus']:
        for control,treatment in [('B1','A1'),('Ref','A1'),('Ref','B1')]:
            result=paired_comparison(quality[(control,kind)],quality[(treatment,kind)],rgb_exception=exception)
            if result['audited_rgb_exceptions']:
                result['strict_pixel_sensitivity']=paired_comparison([r for r in quality[(control,kind)] if r['pair_id']!=exception['pair_id']],
                    [r for r in quality[(treatment,kind)] if r['pair_id']!=exception['pair_id']])
            comparisons[f'{kind}_{treatment}_minus_{control}']=result
    categories=sorted({r['category'] for r in quality[('Ref','plus')]})
    for category in categories:
        a=[r for r in quality[('B1','plus')] if r['category']==category];b=[r for r in quality[('A1','plus')] if r['category']==category]
        result=paired_comparison(a,b,rgb_exception=exception)
        if result['audited_rgb_exceptions']:result['strict_pixel_sensitivity']=paired_comparison([r for r in a if r['pair_id']!=exception['pair_id']],[r for r in b if r['pair_id']!=exception['pair_id']])
        comparisons['category_A1_minus_B1_'+category]=result
    retained=all(comparisons[key]['treatment_minus_control_pp']>=-3 and comparisons[key]['cluster_bootstrap_95_pp'][0]>=-3 for key in ['clean_A1_minus_B1','clean_A1_minus_Ref'])
    efficiency_table=[];baseline=efficiency[(5,10)];baseline_summary=summarize(baseline)
    for (replan,denoise),rows in sorted(efficiency.items()):
        summary=summarize(rows)
        efficiency_table.append({'replan_steps':replan,'num_steps':denoise,**summary,
            'paired_success_vs_R5_N10':paired_comparison(baseline,rows),
            'sampling_speedup_vs_R5_N10':baseline_summary['median_sampling_ms']/summary['median_sampling_ms'],
            'feedback_frequency_ratio_vs_R5_N10':summary['median_simulation_feedback_hz']/baseline_summary['median_simulation_feedback_hz']})
    training=[]
    for model in ['B1','A1']:
        run=Path(state['models'][model]['run']);result=json.loads((run/'result.json').read_text())
        assert result['success'] and result['recorded_training_steps']==5000
        for name in ['reload_cpu.json','actions_sanity.json']:assert json.loads((run/name).read_text())['passed']
        proof=json.loads((run/'update_proof.json').read_text());assert proof['passed']
        params=json.loads((run/'params_initial.json').read_text())
        steps=[json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in steps]==list(range(5000))
        assert all(math.isfinite(v) for r in steps for v in r['metrics'].values())
        training.append({'model':model,'batch':result['batch'],'steps':5000,'seed':42,
            'trainable_parameters':params['trainable_parameters'],'total_parameters':params['total_parameters'],
            'median_step_seconds':result['median_step_seconds_excluding_first_5'],
            'peak_allocator_GiB':result['peak_allocator_bytes_in_use']/1024**3,
            'parameter_update_sample_audit':proof,'checkpoint':state['models'][model]['checkpoint']})
    report={'status':'remote_audit_passed_pending_local_archive_acceptance','coverage':coverage,'quality':table,
        'paired_comparisons':comparisons,'registered_clean_retention_criterion_met':retained,
        'primary_camera_A1_minus_B1':comparisons['category_A1_minus_B1_Camera Viewpoints'],
        'efficiency':efficiency_table,'training':training,'input_pairing_exception':exception,
        'all_original_strict_pairing_gates_passed':False,'analysis_integrity_amendment':amendment,
        'limitations':['One training seed; intervals do not measure training-seed variance.','Heldout perturbation variants share the40 base tasks.','Per-category final sample30.','Concurrent quality-run latency excluded from serial efficiency claims.','Frequency is measured simulator feedback, not physical robot Hz.','Historical default-autotune389/400 and387/400 are separate from the final fixed-kernel common reference; failed exact replay was not relabeled as passed.','No GitHub publication is certified by this analysis.']}
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['model','kind','group','episodes','successes','success_rate','wilson_95']
    with (args.out/'success_rates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(table)
    columns=['replan_steps','num_steps','episodes','successes','success_rate','median_sampling_ms','median_policy_wall_ms','median_request_wall_ms','amortized_action_throughput_hz','median_simulation_feedback_hz','simulation_actions_per_control_second_without_warmup','sampling_speedup_vs_R5_N10','feedback_frequency_ratio_vs_R5_N10']
    with (args.out/'efficiency.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(efficiency_table)
    print(json.dumps({'status':report['status'],'quality_jobs':len(quality),'efficiency_jobs':len(efficiency),'primary_camera':report['primary_camera_A1_minus_B1'],'clean_retention':retained},indent=2))


if __name__=='__main__':main()
