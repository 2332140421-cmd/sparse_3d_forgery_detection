"""Audit already-downloaded official metadata; never infer spatial truth."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

from .data import media_identity


def inventory(metadata):
    info = json.loads((metadata/'dataset-info.json').read_text())
    available = {x['rfilename'] for x in info['siblings']}
    annotations = {}
    with zipfile.ZipFile(metadata/'annot.zip') as z:
        for name in z.namelist():
            if not name.endswith('.txt'): continue
            split, generator = Path(name).stem.split('@')
            for line in z.read(name).decode().splitlines():
                filename, duration, spans = line.rsplit(maxsplit=2)
                key = (split, generator, filename)
                if key in annotations: raise ValueError('DUPLICATE_ANNOTATION')
                intervals = [list(map(float, x.split('='))) for x in spans.split('+')]
                if any(len(x)!=2 or not 0<=x[0]<=x[1]<=float(duration)+.05 for x in intervals):
                    raise ValueError('INVALID_TEMPORAL_INTERVAL')
                annotations[key] = {'duration_s':float(duration), 'intervals_s':intervals}
    rows=[]
    for split in ('train','test'):
        with (metadata/f'{split}.csv').open() as f:
            for record in csv.DictReader(f):
                path=record['file_name']; name=Path(path).name
                if path not in available: raise ValueError('METADATA_MEDIA_MISSING')
                tags=[segment.rsplit('=',1)[-1].split('@') for segment in Path(name).stem.split('+')[1:]]
                if not tags or any(len(tag) not in (4,5) for tag in tags): raise ValueError(f'UNPARSED_IDENTITY:{path}')
                identities={(tag[0],tag[2],tag[-1]) for tag in tags}
                if len(identities)!=1: raise ValueError('AMBIGUOUS_SOURCE')
                dataset,source,generator=next(iter(identities))
                annotation=annotations[(split,generator,name)]
                rows.append({'video_id':media_identity(dataset+':'+source,'fake',path),
                             'source_id':dataset+':'+source,'role':'fake','media_path':path,
                             'official_split':split,'generator':generator,'scene_source':dataset,
                             **annotation})
    counts=lambda key:dict(sorted(Counter(r[key] for r in rows).items()))
    return {'revision':info['sha'],'rows':rows,'summary':{
        'fake_videos':len(rows),'unique_sources':len({r['source_id'] for r in rows}),
        'split':counts('official_split'),'generator':counts('generator'),'scene_source':counts('scene_source'),
        'real_videos_in_AF_repository':0,'published_spatial_annotations_found':0,
        'annotation_semantics':'Temporal manipulated-activity intervals in seconds; no spatial coordinates.',
        'nonmedia_files':sorted(available-{r['media_path'] for r in rows})}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=inventory(a.metadata);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['summary'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
