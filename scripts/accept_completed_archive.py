"""Independently hash, extract and recalculate a downloaded study archive locally."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REMOTE=PurePosixPath('/root/autodl-tmp/vla_reproduction/pi05_robustness')
sys.path.insert(0,str(ROOT/'src'))
from pi05_robustness.statistics import paired_comparison
from evaluate_v3 import plan_tasks,request_seed


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def accept(archive,receipt):
    assert archive.stat().st_size==receipt['bytes'] and digest(archive)==receipt['sha256']
    destination=archive.with_suffix('');destination.resolve().relative_to((ROOT/'evidence').resolve());destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('payload_manifest.json'))
        assert len(manifest)==receipt['files']
        expected={r['path'] for r in manifest}|{'payload_manifest.json'}
        assert len(z.namelist())==len(expected) and set(z.namelist())==expected
        for row in manifest:
            path=PurePosixPath(row['path']);assert not path.is_absolute() and '..' not in path.parts and ':' not in row['path']
            target=destination.joinpath(*path.parts);target.resolve().relative_to(destination.resolve());target.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();n=0
            with z.open(row['path']) as src,target.open('wb') as dst:
                for block in iter(lambda:src.read(4*1024**2),b''):h.update(block);dst.write(block);n+=len(block)
            assert n==row['bytes'] and h.hexdigest()==row['sha256'],row['path']
    def local(path):return destination.joinpath(*PurePosixPath(path).relative_to(REMOTE).parts)
    state=json.loads((destination/'evidence/full_experiment_state.json').read_text(encoding='utf-8'))
    analysis=json.loads((destination/'runs/final_analysis/results.json').read_text(encoding='utf-8'))
    assert len(state['evaluations'])==11 and analysis['status']=='remote_audit_passed_pending_local_archive_acceptance'
    for name in ['protocol.json','eval_subset.json','eval_subset.heldout.json','training_plan.frozen.json','augmentation_registration.json','analysis_registration.json','inference_reproducibility_registration.json','evaluation_runtime_amendment.json']:
        assert digest(destination/'configs'/name)==digest(ROOT/'configs'/name),name
    registration=json.loads((destination/'configs/inference_reproducibility_registration.json').read_text(encoding='utf-8'))
    for record in registration['caches'].values():assert digest(local(record['path']))==record['sha256']
    exception=json.loads((destination/'evidence/render_rounding_exception.json').read_text(encoding='utf-8'))
    amendment=json.loads((destination/'configs/analysis_integrity_amendment.json').read_text(encoding='utf-8'))
    assert digest(destination/'evidence/render_rounding_exception.json')==amendment['exception_proof_sha256']
    assert not exception['original_strict_pixel_gate_passed'] and exception['original_records_unchanged']
    import numpy as np
    images=[np.load(local(path),allow_pickle=False) for path in exception['recovered_files']]
    for kind,expected in [('raw',9),('model_input',10)]:
        a,b=[image[kind] for image in images];difference=np.abs(a.astype(np.int16)-b.astype(np.int16))
        assert np.count_nonzero(difference)==expected and difference.max()==1
        assert [hashlib.sha256(x.tobytes()).hexdigest() for x in [a,b]]==exception['metrics'][kind]['sha256']
    jobs=[];quality={};efficiency={}
    for job in state['evaluations']:
        out=local(job['out']);directories=sorted((out/'episodes').glob('*/run.json')) if not job['serial'] else [out/'episodes/run.json'];rows=[]
        for metadata_path in directories:
            metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
            server=metadata['server'];cache=registration['caches'][server['config']+'_N'+str(job['num_steps'])]
            assert server['inference_autotune_cache_sha256']==cache['sha256']
            assert '--xla_gpu_require_complete_aot_autotune_results=true' in server['xla_flags']
            records=[json.loads(x) for x in metadata_path.with_name('episodes.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(records)==metadata['expected_episodes'] and all(r['status']=='complete' for r in records)
            rows.extend(records)
        expected_plan=plan_tasks(job['kind'],'libero_spatial' if job['serial'] else 'all',10 if job['kind']=='clean' else 1,'heldout')
        identity=lambda r:(r['suite'],r['task_name'],r['init_state_index'])
        assert len(rows)==len(expected_plan) and {identity(r) for r in rows}=={identity(r) for r in expected_plan}
        for row in rows:
            n=row['environment_steps'];assert len(row['actions'])==len(row['rewards'])==len(row['done_signals'])==n
            assert type(row['success']) is bool and row['success']==any(row['done_signals'])
            assert all(len(a)==7 and all(math.isfinite(x) for x in a) for a in row['actions'])
            assert all(math.isfinite(x) for x in row['rewards'])
            assert local(row['video']).is_file()
            assert [t['environment_step'] for t in row['inference']]==list(range(0,n,job['replan_steps']))
            for t in row['inference']:assert t['request_seed_uint32']==request_seed(row['flow_noise_identity'],t['environment_step'],row['seed'])
        result=json.loads((out/'result.json').read_text(encoding='utf-8'));assert result['status']=='complete' and result['successes']==sum(r['success'] for r in rows)
        jobs.append({**job,'episodes':len(rows),'successes':sum(r['success'] for r in rows)})
        if job['serial']:efficiency[(job['replan_steps'],job['num_steps'])]=rows
        else:quality[(job['model'],job['kind'])]=rows
    for kind in ['clean','plus']:
        for control,treatment in [('B1','A1'),('Ref','A1'),('Ref','B1')]:
            recalculated=paired_comparison(quality[(control,kind)],quality[(treatment,kind)],rgb_exception=exception)
            if recalculated['audited_rgb_exceptions']:
                recalculated['strict_pixel_sensitivity']=paired_comparison([r for r in quality[(control,kind)] if r['pair_id']!=exception['pair_id']],[r for r in quality[(treatment,kind)] if r['pair_id']!=exception['pair_id']])
            assert recalculated==analysis['paired_comparisons'][f'{kind}_{treatment}_minus_{control}']
    for row in analysis['efficiency']:
        assert paired_comparison(efficiency[(5,10)],efficiency[(row['replan_steps'],row['num_steps'])])==row['paired_success_vs_R5_N10']
    result={'passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':receipt['sha256'],
        'verified_payload_files':len(manifest),'episodes_checked':sum(r['episodes'] for r in jobs),'jobs':jobs,
        'local_paired_statistics_match':True,'manual_video_review_complete':False,'github_published':False,
        'all_original_strict_pairing_gates_passed':False,'bounded_rgb_exception_count':1,
        'bounded_rgb_exception_independently_verified':True,
        'scope':'Full archive hashes, coverage, raw action/reward/done/seed checks and independent local paired-statistic recalculation. Video content review and publication remain separate.'}
    (ROOT/'evidence/completed_study_local_acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# 实验真实结果（自动验收）','','全部训练与评测任务已结束，原始证据已下载并通过本地哈希、覆盖、动作、随机数及配对统计复算。视频内容人工复核和GitHub发布尚未完成。','','| 模型 | 干净LIBERO（400回合） | 独立Plus留出（210回合） |','|---|---:|---:|']
    for model in ['Ref','B1','A1']:
        values=[f"{sum(r['success'] for r in quality[(model,k)])}/{len(quality[(model,k)])}" for k in ['clean','plus']]
        lines.append('| '+model+' | '+' | '.join(values)+' |')
    primary=analysis['primary_camera_A1_minus_B1'];overall=analysis['paired_comparisons']['plus_A1_minus_B1']
    lines+=['',f"主终点（视角，30回合）：A1−B1为{primary['treatment_minus_control_pp']:.2f}个百分点，任务簇bootstrap95%区间{primary['cluster_bootstrap_95_pp']}。",f"整体Plus留出：A1−B1为{overall['treatment_minus_control_pp']:.2f}个百分点；本次实验未观察到增强优于默认继续微调的整体收益。",f"预先登记的干净性能保持标准是否满足：{analysis['registered_clean_retention_criterion_met']}。",'', '完整210回合均保留。B1有1个光照初始帧与另外两模型不同：原图9个颜色通道、模型输入10个颜色通道相差1灰度级。原严格像素门槛没有全部通过；该例外通过原帧哈希重建并独立复核，涉及的统计另提供剔除该对后的209回合严格敏感性分析。视角主终点、干净和效率配对无需例外。','', '单训练seed=42；独立扰动变体仍属于相同40个基础任务。上述区间不反映训练种子方差。不得将动作吞吐称为真实机器人闭环频率。','','完整表、区间及三幅科学图位于下载证据中的runs/final_analysis/。']
    (ROOT/'最终实验实测结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('receipt',type=Path);args=p.parse_args()
    print(json.dumps(accept(args.archive,json.loads(args.receipt.read_text(encoding='utf-8'))),indent=2))
