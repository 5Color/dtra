"""Run the uploaded recorder with fixed episodes and every simulator step.

The benchmark and upstream repositories remain untouched. Only the in-memory
frame capture condition and recorder presentation are changed.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path


def full_frame_source(source: str) -> str:
    old = 'record_hud_frames is not None and (step_i % 3 == 0 or solved or step_i == n_steps - 1)'
    assert source.count(old) == 1, 'Unexpected benchmark capture implementation'
    return source.replace(old, 'record_hud_frames is not None')


def unedited_main_source(source: str) -> str:
    replacements = {
        'for ep_id in tinfo["eval_eps"]:': 'for ep_id in tinfo["eval_eps"][:1]:',
        '                frames.extend([frames[-1]] * 6)\n': '',
        'duration=90, loop=0, optimize=True': 'duration=round(1000 / recording_fps), loop=0, optimize=False',
        'all_hud_frames, fps=12)': 'all_hud_frames, fps=recording_fps, macro_block_size=1)',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, f'Unexpected recorder implementation: {old}'
        source = source.replace(old, new)
    return source


def main() -> None:
    import run_robocasa_continual_benchmark as benchmark
    import record_robocasa_hud as recorder

    ns = dict(vars(benchmark))
    capture_source = full_frame_source(inspect.getsource(benchmark.rollout_spline_episode))
    exec(compile(capture_source, '<full-frame-capture>', 'exec'), ns)
    full_rollout = ns['rollout_spline_episode']
    recording = {
        'episode_selection': 'First held-out episode in cache order, fixed before observing outcomes',
        'capture': 'Every simulator step; no dropped steps, repeated hold frames, or success selection',
        'evaluation': 'Original policy, actions, horizon and env._check_success() unchanged',
        'drift_metric_limit': 'DTRA parameter_drift() returns a structural constant; not an independent weight comparison',
        'episodes': [],
    }
    chosen_episodes = {}
    original_load = recorder.load_or_build_task_cache
    recorder_ns = dict(vars(recorder))
    recorder_ns['recording_fps'] = None

    def load_fixed_plan(*args, **kwargs):
        bundle = original_load(*args, **kwargs)
        chosen_episodes.update({name: task['eval_eps'][0] for name, task in bundle['tasks'].items()})
        return bundle

    def capture(*args, **kwargs):
        env, _, policy, _, task = args[:5]
        fps = float(env.control_freq)
        assert recorder_ns['recording_fps'] in (None, fps), 'Mixed control frequencies'
        recorder_ns['recording_fps'] = fps
        records = kwargs['record_hud_frames']
        assert not records
        result = full_rollout(*args, **kwargs)
        steps = [int(r['step']) for r in records]
        assert steps == list(range(1, len(records) + 1)), 'Missing or duplicate simulator steps'
        assert result['solved'] or steps[-1] == records[-1]['max_steps'], 'Truncated episode'
        start = sum(e['frame_count'] for e in recording['episodes'])
        recording['episodes'].append({
            'task': task, 'episode_id': int(chosen_episodes[task]),
            'frame_start_zero_based': start, 'frame_count': len(records),
            'control_frequency_hz': fps, 'result': result,
            'steps': steps,
        })
        print(f'UNEDITED {task} ep{chosen_episodes[task]}: steps=1..{len(records)}, solved={result["solved"]}, fps={fps}', flush=True)
        return result

    recorder_ns['load_or_build_task_cache'] = load_fixed_plan
    recorder_ns['rollout_spline_episode'] = capture
    source = unedited_main_source(inspect.getsource(recorder.main))
    exec(compile(source, '<fixed-episode-recorder>', 'exec'), recorder_ns)
    recorder_ns['main']()
    out_dir = Path('/home/team03/dtra_a6000_lab/robocasa_lab/dtra_v4')
    if '--out_dir' in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index('--out_dir') + 1])
    recording['frame_count'] = sum(e['frame_count'] for e in recording['episodes'])
    (out_dir / 'recording_provenance.json').write_text(json.dumps(recording, indent=2) + '\n')
    print(f'Validated {recording["frame_count"]} consecutive episode frames.', flush=True)


if __name__ == '__main__':
    main()
