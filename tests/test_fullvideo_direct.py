"""Research contracts: exact attention, all-frame gradients and frozen input geometry."""
import copy
import unittest
import torch
from research_tools.xyz_pilot.fullvideo_direct_model import (
    DirectLayer, FullVideoDirect, prepare_frame, video_loss, SCALE)
from research_tools.xyz_pilot.joint_model import JointLayer


def frames():
    result=[]
    for i in range(3):
        xyz=torch.zeros(6,8,3);xyz[...,0]=torch.arange(8)[None]*SCALE*.3
        xyz[...,2]=SCALE*(1+i*.3)
        valid=torch.ones(6,8,dtype=torch.bool);valid[0,0]=False
        result.append(prepare_frame(dict(xyz_provider=xyz,valid=valid,
            work_hw=[6,8],dino_padded_hw=[14,14],patch=torch.randn(1024,1,1),
            detail=torch.randn(32,4,4),z0=torch.randn(6,8),pts=i*.1,
            gt_work=torch.ones(6,8,dtype=torch.bool),supervised=i!=1)))
    return result


class FullVideoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5)

    def test_sdpa_matches_old_joint_outputs_and_all_gradients(self):
        old=JointLayer().double();new=DirectLayer().double();new.load_state_dict(old.state_dict())
        x=torch.randn(9,128,dtype=torch.double,requires_grad=True)
        p=torch.randn_like(x,requires_grad=True)
        y=old.attend(x,p);y.square().sum().backward()
        gx,gp=x.grad.clone(),p.grad.clone();x.grad=None;p.grad=None
        z=new.attend(x,p);z.square().sum().backward()
        torch.testing.assert_close(y,z,atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(x.grad,gx,atol=1e-10,rtol=1e-9)
        torch.testing.assert_close(p.grad,gp,atol=1e-10,rtol=1e-9)
        for a,b in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-9,rtol=1e-8)

    def test_query_chunks_keep_complete_keys_and_gradients(self):
        a=DirectLayer().double();b=DirectLayer(3).double();b.load_state_dict(a.state_dict())
        x=torch.randn(11,128,dtype=torch.double,requires_grad=True);p=torch.randn_like(x)
        ya=a.attend(x,p);ga=torch.autograd.grad(ya.square().sum(),(x,*a.parameters()))
        yb=b.attend(x,p);gb=torch.autograd.grad(yb.square().sum(),(x,*b.parameters()))
        torch.testing.assert_close(ya,yb)
        for u,v in zip(ga,gb):torch.testing.assert_close(u,v)

    def test_future_dependency_exists_only_for_full_video(self):
        model=FullVideoDirect(activation_checkpoint=False)
        for mode in ['FRAME_ONLY','FULL_VIDEO']:
            x=torch.randn(6,128,requires_grad=True);p=torch.randn_like(x)
            r=model.relate(x,p,[2,2,2],mode)
            g=torch.autograd.grad(r[:2].square().sum(),x)[0][2:]
            self.assertEqual(bool(g.abs().sum()>0),mode=='FULL_VIDEO')

    def test_initial_outputs_all_equal_cached_z0_and_excluded_loss(self):
        fs=frames();model=FullVideoDirect(activation_checkpoint=False)
        for mode in ['FRAME_ONLY','FULL_VIDEO']:
            out=model(fs,mode)
            self.assertEqual(len(out),len(fs))
            for z,f in zip(out,fs):torch.testing.assert_close(z,f['z0'],rtol=0,atol=0)
            loss,_=video_loss(out,fs)
            other=copy.deepcopy(fs);other[1]['gt_work'].zero_()
            torch.testing.assert_close(loss,video_loss(out,other)[0])

    def test_checkpoint_binds_each_frame_and_preserves_parameter_gradients(self):
        fs=frames();a=FullVideoDirect(activation_checkpoint=False)
        torch.nn.init.normal_(a.correction[-1].weight,std=.01)
        b=copy.deepcopy(a);b.activation_checkpoint=True
        for mode in ['FRAME_ONLY','FULL_VIDEO']:
            a.zero_grad();b.zero_grad()
            ya=a(fs,mode);yb=b(fs,mode)
            video_loss(ya,fs)[0].backward();video_loss(yb,fs)[0].backward()
            for u,v in zip(ya,yb):torch.testing.assert_close(u,v)
            for (n,u),(_,v) in zip(a.named_parameters(),b.named_parameters()):
                self.assertIsNotNone(u.grad,n);self.assertIsNotNone(v.grad,n)
                torch.testing.assert_close(u.grad,v.grad,rtol=1e-4,atol=1e-6)

    def test_environment_failure_retains_report_and_python_exit(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from research_tools.xyz_pilot.fullvideo_direct_probe import main
        with tempfile.TemporaryDirectory() as d:
            with patch('sys.argv',['probe','--output',d]), patch('torch.cuda.is_available',return_value=False):
                with self.assertRaises(SystemExit) as result:main()
            self.assertEqual(result.exception.code,1)
            value=json.loads((Path(d)/'cases/environment/result.json').read_text())
            self.assertEqual(value['stage'],'environment')
            self.assertEqual(value['exit_code'],1)
            self.assertTrue((Path(d)/'report.md').exists())

    def test_existing_result_is_not_overwritten_by_new_failure(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from research_tools.xyz_pilot.fullvideo_direct_probe import main
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'cases/environment/result.json'
            target.parent.mkdir(parents=True)
            original=dict(case='environment',status='FAILED',stage='original_failure')
            target.write_text(json.dumps(original))
            with patch('sys.argv',['probe','--output',d]), patch('torch.cuda.is_available',return_value=False):
                with self.assertRaises(SystemExit):main()
            self.assertEqual(json.loads(target.read_text()),original)
            self.assertEqual(len(list((Path(d)/'invocations').glob('failure_*.json'))),1)

    def test_fp32_routing_single_scale_and_invalid_bypass(self):
        fs=frames();f=fs[0]
        self.assertEqual(len(f['points']),47)
        self.assertEqual(f['keys'].shape[1],3)
        expected=torch.floor((f['xyz_provider']/SCALE).reshape(-1,3)[f['points']]/.5).long()
        self.assertTrue(torch.equal(f['keys'][f['index']],expected))
        for f in fs:f['valid'].zero_()
        fs=[prepare_frame({k:v for k,v in f.items() if k not in ['xyz_normalized','points','keys','index']}) for f in fs]
        model=FullVideoDirect(activation_checkpoint=False)
        out=model(fs,'FULL_VIDEO')
        for z,f in zip(out,fs):torch.testing.assert_close(z,f['z0'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
