import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from research_tools.xyz_pilot import change_relation as entry
from research_tools.xyz_pilot import change_relation_report
class DriverTest(unittest.TestCase):
    def test_ten_epoch_driver_exposures_logging_checkpoint_and_completion(self):
        protocol=entry.olddata.read(entry.SOURCE/'protocol.json');ids=[r['video_id'] for r in protocol['rows'] if r['split']=='train']
        cache=SimpleNamespace(load=lambda row,c:([dict(keys=[0],supervised=entry.olddata.supervised(row,i)) for i in range(row['actual_frames'])],0.))
        cp={};messages=[]
        def checkpoint(path,model,opt,identity,c,e,pos,records):
            cp.update(epoch=e,position=pos,updates=len(records),records=records);path.parent.mkdir(exist_ok=True);path.touch();return cp
        def fake_update(m,o,fs,c,emit):return dict(loss=1.,pixel_BCE=.3,soft_Dice=.6,block_BCE=1.,seconds=.01,supervised_frames=sum(f['supervised'] for f in fs),peak_allocated_bytes=0,peak_reserved_bytes=0)
        metrics={k:.2 for k in change_relation_report.KEYS}
        def evaluate(m,cache,rows,c,emit,categories=None,monitor=None,capture=None):return dict(frames=[None]*(len(monitor) if monitor else 5152),macro=metrics,groups={'新增15开发来源':metrics},sources=[])
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);entry.save(out/'orders.json',[ids]*10);entry.save(out/'monitor.json',[[ids[0],i] for i in range(32)]);entry.save(out/'categories.json',{})
            model=SimpleNamespace(cuda=lambda:None)
            model.cuda=lambda:model
            opt=SimpleNamespace(param_groups=[{},{}])
            with patch.object(entry,'initial',return_value=model),patch.object(entry,'optimizer',return_value=opt),patch.object(entry,'buffers',return_value='fixed'),patch.object(entry,'Cache',return_value=cache),patch.object(entry,'update',side_effect=fake_update),patch.object(entry,'save_checkpoint',side_effect=checkpoint),patch.object(entry,'restore_checkpoint',return_value=cp),patch.object(entry.torch,'load',side_effect=lambda *a,**k:cp),patch.object(entry.olddata,'atomic_torch',side_effect=lambda p,c:p.touch()),patch.object(entry,'evaluate',side_effect=evaluate),patch.object(entry,'persist_eval',side_effect=lambda p,e:entry.save(p,e)),patch.object(change_relation_report,'report'):
                entry.run_condition(out,protocol,{}, {},'A_CURRENT_XY',False,lambda stage,**kw:messages.append((stage,kw)))
            d=json.loads((out/'A_CURRENT_XY/complete.json').read_text());self.assertEqual(d['updates'],1780);self.assertEqual(d['supervised_presentations'],228780)
            self.assertEqual(sum(stage=='train_progress' for stage,_ in messages),1780)
            self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6)
            self.assertEqual(cp['epoch'],11);self.assertEqual(cp['position'],0)
if __name__=='__main__':unittest.main()
