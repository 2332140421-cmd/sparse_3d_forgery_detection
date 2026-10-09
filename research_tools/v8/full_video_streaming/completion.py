"""Close the progress view after successful pipeline exit (no numerical work)."""
import argparse
import json
import os
import time
from pathlib import Path


def finalize(out):
    path=out/'final_status.json';status=json.loads(path.read_text())
    if status['status']!='COMPLETE': raise ValueError('CANNOT_FINALIZE_INCOMPLETE')
    if status['completion_scope']!='two_video_engineering_pilot' or status['completed_video_count']!=2:
        raise ValueError('COMPLETION_SCOPE')
    n=status['decoded_frame_count']
    if n!=status['frontend_frame_count'] or n!=status['response_frame_count']: raise ValueError('COMPLETION_COUNTS')
    status.update(training_started=False,full_dataset_processed=False)
    progress={'status':'COMPLETE','stage':'report','completed_video_count':2,'completed_frames':n,
              'total_frames':n,'completion_scope':'two_video_engineering_pilot','updated_unix':time.time()}
    for target,value in ((path,status),(out/'progress.json',progress)):
        tmp=target.with_name(target.name+'.completion.tmp')
        with tmp.open('w') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,target)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args,_=p.parse_known_args();finalize(Path(args.output))
