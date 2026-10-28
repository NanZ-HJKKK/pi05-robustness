"""Audit the exact two initial frames behind the sole RGB pairing exception."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    state=json.loads((ROOT/'evidence/full_experiment_state.json').read_text())
    name='KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_4'
    rows=[]
    for model in ['Ref','B1','A1']:
        job=next(j for j in state['evaluations'] if j['model']==model and j['kind']=='plus')
        row=next(r for p in (Path(job['out'])/'episodes').rglob('episodes.jsonl') for r in map(json.loads,p.read_text().splitlines()) if r['task_name']==name)
        assert row['success'] is True;rows.append(row)
    fields=['pair_id','initial_state_sha256','settled_sim_state_sha256','initial_wrist_rgb_sha256','bddl_sha256','prompt','seed']
    assert all(all(r[f]==rows[0][f] for f in fields) for r in rows)
    hashes=sorted({r['initial_main_rgb_sha256'] for r in rows});assert len(hashes)==2
    recovered={}
    for result in (ROOT/'runs/render_hash_exception').glob('*/result.json'):
        data=json.loads(result.read_text());assert data['settled_state_sha256']==rows[0]['settled_sim_state_sha256']
        for digest in hashes:
            file=result.parent/(digest+'.npz')
            if file.exists():recovered.setdefault(digest,file)
    assert set(recovered)==set(hashes)
    arrays=[np.load(recovered[h]) for h in hashes]
    for h,a in zip(hashes,arrays):assert hashlib.sha256(a['raw'].tobytes()).hexdigest()==h
    metrics={}
    for kind in ['raw','model_input']:
        a,b=[x[kind] for x in arrays];assert a.shape==b.shape and a.dtype==b.dtype==np.uint8
        difference=np.abs(a.astype(np.int16)-b.astype(np.int16));coordinates=np.argwhere(difference)
        assert difference.max()==1
        metrics[kind]={'shape':list(a.shape),'total_channels':int(a.size),'changed_channels':int(np.count_nonzero(difference)),
            'max_uint8_difference':int(difference.max()),'coordinates':coordinates.tolist(),
            'values_at_changed_channels':[[int(a[tuple(c)]),int(b[tuple(c)])] for c in coordinates],
            'sha256':[hashlib.sha256(x.tobytes()).hexdigest() for x in [a,b]]}
    assert metrics['raw']['changed_channels']==9 and metrics['model_input']['changed_channels']==10
    proof={'status':'audited_bounded_rgb_exception','pair_id':rows[0]['pair_id'],'field':'initial_main_rgb_sha256','raw_hashes':hashes,
        'metrics':metrics,'recovered_files':[str(recovered[h]) for h in hashes],
        'physical_state_and_all_other_pairing_fields_equal':True,'all_three_observed_successes':True,
        'original_strict_pixel_gate_passed':False,'model_input_byte_identical':False,'original_records_unchanged':True,
        'interpretation':'The exact original frames were recovered by hash. Full210 counts remain unchanged; comparisons applying this sole exception must disclose it and also report strict209 sensitivity. No generic pixel tolerance is authorized.'}
    path=ROOT/'evidence/render_rounding_exception.json';path.write_text(json.dumps(proof,indent=2)+'\n')
    amendment={'timing':'Post hoc data-quality investigation after all scored outcomes were available','strict_pixel_pairing_deviation_count':1,
        'exception_proof_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'rule':'Only this exact pair ID and these two recovered raw hashes may bypass the main-RGB equality gate; all other fields stay exact. Report full210 and strict209 sensitivity, retain all original outcomes.',
        'primary_camera_clean_and_efficiency_inputs':'All exact; no exception applies','weights_or_episode_scores_changed':False}
    (ROOT/'configs/analysis_integrity_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
    state.setdefault('validation_runs',[]).append({'out':str(ROOT/'runs/render_hash_exception'),'reason':'Exact reconstruction of the sole initial RGB exception; no scored policy episodes'})
    tmp=ROOT/'evidence/full_experiment_state.writing';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ROOT/'evidence/full_experiment_state.json')
    print(json.dumps({'raw_changed_channels':9,'model_changed_channels':10,'max_uint8_difference':1,'strict_pixel_gate_passed':False},indent=2))


if __name__=='__main__':main()
