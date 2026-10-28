"""Recompute every published success comparison from compressed episode records."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pi05_robustness.statistics import paired_comparison, summarize


def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        target = (ROOT / item['path']).resolve()
        target.relative_to(ROOT.resolve())
        assert target.stat().st_size == item['bytes'], item['path']
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item['sha256'], item['path']
    report = json.loads((ROOT / 'results/results.json').read_text(encoding='utf-8'))
    index = json.loads((ROOT / 'results/episode_index.json').read_text(encoding='utf-8'))
    exception = json.loads((ROOT / 'evidence/render_rounding_exception.json').read_text(encoding='utf-8'))
    quality, efficiency = {}, {}
    total = 0
    for job in index:
        with gzip.open(ROOT / job['records'], 'rt', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == job['episodes']
        assert sum(row['success'] for row in rows) == job['successes']
        for row in rows:
            n = row['environment_steps']
            assert len(row['actions']) == len(row['rewards']) == len(row['done_signals']) == n
            assert row['success'] == any(row['done_signals'])
            assert all(len(action) == 7 and all(math.isfinite(v) for v in action) for action in row['actions'])
            assert all(math.isfinite(v) for v in row['rewards'])
        total += len(rows)
        if job['serial']:
            efficiency[job['replan_steps'], job['num_steps']] = rows
        else:
            quality[job['model'], job['kind']] = rows
        for record in report['quality']:
            if (record['model'], record['kind']) == (job['model'], job['kind']) and not job['serial']:
                selected = rows if record['group'] == 'overall' else [
                    r for r in rows if r['suite' if job['kind'] == 'clean' else 'category'] == record['group']]
                assert summarize(selected) == {k: v for k, v in record.items() if k not in ('model', 'kind', 'group')}
    comparisons = report['paired_comparisons']
    for kind in ('clean', 'plus'):
        for control, treatment in [('B1', 'A1'), ('Ref', 'A1'), ('Ref', 'B1')]:
            a, b = quality[control, kind], quality[treatment, kind]
            value = paired_comparison(a, b, rgb_exception=exception)
            if value['audited_rgb_exceptions']:
                value['strict_pixel_sensitivity'] = paired_comparison(
                    [r for r in a if r['pair_id'] != exception['pair_id']],
                    [r for r in b if r['pair_id'] != exception['pair_id']])
            assert value == comparisons[f'{kind}_{treatment}_minus_{control}']
    for key, expected in comparisons.items():
        if not key.startswith('category_A1_minus_B1_'):
            continue
        category = key.removeprefix('category_A1_minus_B1_')
        a = [r for r in quality['B1', 'plus'] if r['category'] == category]
        b = [r for r in quality['A1', 'plus'] if r['category'] == category]
        value = paired_comparison(a, b, rgb_exception=exception)
        if value['audited_rgb_exceptions']:
            value['strict_pixel_sensitivity'] = paired_comparison(
                [r for r in a if r['pair_id'] != exception['pair_id']],
                [r for r in b if r['pair_id'] != exception['pair_id']])
        assert value == expected, key
    for item in report['efficiency']:
        rows = efficiency[item['replan_steps'], item['num_steps']]
        assert paired_comparison(efficiency[5, 10], rows) == item['paired_success_vs_R5_N10']
        summary = summarize(rows)
        assert all(item[k] == v for k, v in summary.items())
        baseline = summarize(efficiency[5, 10])
        assert item['sampling_speedup_vs_R5_N10'] == baseline['median_sampling_ms'] / summary['median_sampling_ms']
        assert item['feedback_frequency_ratio_vs_R5_N10'] == summary['median_simulation_feedback_hz'] / baseline['median_simulation_feedback_hz']
    for model in ('B1', 'A1'):
        steps = [json.loads(line) for line in (ROOT / 'results/training' / model / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['step'] for r in steps] == list(range(5000))
        assert all(math.isfinite(v) for r in steps for v in r['metrics'].values())
    assert total == 2330 and len(index) == 11
    print(json.dumps({'passed': True, 'episodes': total, 'jobs': len(index),
                      'paired_statistics_recomputed': True, 'github_published': False}))


if __name__ == '__main__':
    main()
