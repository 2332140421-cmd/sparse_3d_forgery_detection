"""Publish this experiment's completed small report, never restart computation."""
import argparse
import subprocess
from pathlib import Path

from research_tools.v8.component_global_context_pilot.run import save_json
from research_tools.v8.full_coverage_observation_field.common import read_json, sha256_file


def publish(out):
    repo=Path(__file__).resolve().parents[3]
    status=read_json(out/'final_status.json')
    if status.get('status')!='COMPLETE' or status.get('completed_videos')!=272 or status.get('completed_models')!=6:
        raise ValueError(f"EXPERIMENT_NOT_COMPLETE:{status.get('status')}")
    manifest=read_json(out/'video_manifest.json')['rows'];frames=0
    for row in manifest:
        s=read_json(out/'responses'/row['video_id']/'summary.json')
        if s['status']!='COMPLETE' or s['video_sha256']!=row['video_sha256'] or len(s['video_logits'])!=6 or s['decoded_frames']!=s['response_frames_per_model']:
            raise ValueError('FINAL_VIDEO_COVERAGE')
        frames+=s['decoded_frames']
    if frames!=status['decoded_frames']: raise ValueError('FINAL_FRAME_COUNT')
    protocol=read_json(out/'protocol.json')
    for path,expected in protocol['frozen_files'].items():
        if sha256_file(path)!=expected: raise ValueError(f'FROZEN_FILE_CHANGED:{path}')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=repo,text=True).strip()
    if git('branch','--show-current')!='v8-full-video-global-context': raise ValueError('PUBLISH_BRANCH_CHANGED')
    target='docs/experiments/v8-full-video-global-context.md'
    if git('diff','--name-only','--',target) or git('diff','--cached','--name-only'):
        raise ValueError('PUBLISH_WOULD_MIX_EXISTING_CHANGES')
    text=(out/'report.md').read_text()
    text+='\n完整产物：`'+str(out)+'`。必要测试与原窗口/连续输入数值验证见validation.json及native_continuous_validation.json。\n'
    (repo/target).write_text(text)
    subprocess.run(['git','diff','--check'],cwd=repo,check=True)
    subprocess.run(['git','add','--',target],cwd=repo,check=True)
    if git('diff','--cached','--name-only'):
        subprocess.run(['git','commit','-m','Report completed frozen global-context full-video evaluation'],cwd=repo,check=True)
    subprocess.run(['git','push','origin','v8-full-video-global-context'],cwd=repo,check=True)
    save_json(out/'git_delivery.json',{'status':'PUSHED','head':git('rev-parse','HEAD'),'report':target,
              'videos':len(manifest),'frames':frames,'note':'Report only; no data/models staged'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    try: publish(out)
    except BaseException as e:
        save_json(out/'git_delivery.json',{'status':'NOT_PUSHED','error':f'{type(e).__name__}: {e}'})
        raise
