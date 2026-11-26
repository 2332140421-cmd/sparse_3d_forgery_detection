"""Research-critical routing, cross-time dependence, loss and decoder contracts."""
import copy,sys,tempfile,unittest
from pathlib import Path
import torch
from research_tools.xyz_pilot.change_relation_model import (AINPAINT,ChangeRelation,slots,neighbors,organize,block_targets,block_loss,video_loss)
from research_tools.xyz_pilot.model import route
from research_tools.xyz_pilot.fullvideo_matched import save_checkpoint,restore_checkpoint
from research_tools.xyz_pilot.fullvideo_direct_probe import state_hash
from research_tools.xyz_pilot.change_relation_report import report

def head():
    sys.path.insert(0,str(AINPAINT)) if str(AINPAINT) not in sys.path else None
    from ainpaint.modules.context_decoder import HighResContextHead
    return HighResContextHead(1024,1,norm='BN')

def fixture():
    h,w=13,15;y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    xyz=torch.stack((x.float()/8,y.float()/8,torch.ones_like(x).float()),-1)
    points,keys,index=route(xyz,torch.ones(h,w,dtype=torch.bool),.5)
    return dict(work_hw=[h,w],dino_padded_hw=[14,28],patch=torch.randn(1024,1,2),points=points,keys=keys,index=index,xyz_normalized=xyz,
                gt_work=torch.zeros(h,w,dtype=torch.bool),supervised=True,pts=0.,frame_index=0,native_GT_pixels=0)

class ChangeTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(3)
    def test_xy_edges_common_support_and_no_gt_or_xyz_numeric_input(self):
        f=fixture();a=organize([f],'A_CURRENT_XY')[0];other=copy.deepcopy(f);other['gt_work'].fill_(True);other['xyz_normalized'].fill_(99)
        b=organize([other],'A_CURRENT_XY')[0]
        self.assertEqual(len(a['keys']),2);self.assertEqual(int(a['counts'].sum()),195)
        points=f['points'];coords=torch.stack(((points%15)//14,(points//15)//14,torch.zeros_like(points)),1)
        keys,index=torch.unique(coords,dim=0,sorted=True,return_inverse=True)
        torch.testing.assert_close(a['keys'],keys,rtol=0,atol=0);torch.testing.assert_close(a['index'],index,rtol=0,atol=0)
        for k in ['keys','index','point_coordinates','centers','counts']:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        c=organize([f],'C_FULL_XYZ')[0]
        for k in ['points','keys','index']:torch.testing.assert_close(c[k],f[k])
        self.assertEqual(int(c['counts'].sum()),195)
    def test_slots_nearest_actual_pts_and_bidirectional(self):
        self.assertEqual(slots([0.,.2,.4,.8,1.],2),[1,3])
        self.assertEqual(slots([0.],0),[0])
        s=slots([0.,.2,.4,.6,.8],2);self.assertTrue(any(j<2 for j in s) and any(j>2 for j in s));self.assertEqual(len(s),len(set(s)))
        self.assertTrue(all(j!=0 for j in slots([0.,.2,.4],0)))
    def test_neighbors_exclude_self_and_keep_all_available(self):
        k=torch.tensor([[0,0,0],[1,0,0],[2,0,0]]);n=neighbors(k)
        for i,ids in enumerate(n):self.assertNotIn(i,ids.tolist());self.assertEqual(len(ids),2)
        self.assertEqual(neighbors(k[:1]).tolist(),[[0]])
    def test_soft_block_loss_is_exact_expanded_pixel_bce(self):
        f=dict(points=torch.arange(4),index=torch.tensor([0,0,1,1]),counts=torch.tensor([2,2]),gt_work=torch.tensor([[1,0,1,1]],dtype=torch.bool))
        y,n=block_targets(f,'cpu');torch.testing.assert_close(y,torch.tensor([.5,1.]))
        z=torch.tensor([-.3,.8],requires_grad=True);a=block_loss(z,y,n)
        b=torch.nn.functional.binary_cross_entropy_with_logits(z[f['index']],f['gt_work'].flatten().float());torch.testing.assert_close(a,b)
        g=torch.autograd.grad(a,z,retain_graph=True)[0];h=torch.autograd.grad(b,z)[0];torch.testing.assert_close(g,h)
    def test_zero_fusion_official_parity_padding_and_fixed_bn(self):
        m=ChangeRelation(head(),activation_checkpoint=False).eval();f=organize([fixture()],'A_CURRENT_XY')[0];r=torch.ones(len(f['keys']),128)
        canvas=m.relation_map(f,r)
        self.assertEqual(canvas.shape,(1,128,14,28));self.assertEqual(float(canvas[:,:,13:,:].abs().sum()),0);self.assertEqual(float(canvas[:,:,:,15:].abs().sum()),0)
        with torch.no_grad():
            a=m.decode(f,r);b=m.head(f['patch'][None],target_size=[14,28])[0,0,:13,:15]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        m.train();self.assertTrue(m.head.dropout.training)
        for b in m.head.modules():
            if isinstance(b,torch.nn.BatchNorm2d):self.assertFalse(b.training);self.assertTrue(b.weight.requires_grad)
    def test_a_cannot_read_other_frames_b_does(self):
        fs=[dict(fixture(),pts=t,frame_index=i) for i,t in enumerate([0.,.2,.4])]
        m=ChangeRelation(head(),activation_checkpoint=False).eval()
        a=organize(fs,'A_CURRENT_XY');b=organize(fs,'B_FULL_XY');changed=copy.deepcopy(fs);changed[2]['patch'].add_(2)
        with torch.no_grad():
            za=m(a,'A_CURRENT_XY')[1][0];zb=m(b,'B_FULL_XY')[1][0]
            aa=m(organize(changed,'A_CURRENT_XY'),'A_CURRENT_XY')[1][0];bb=m(organize(changed,'B_FULL_XY'),'B_FULL_XY')[1][0]
        torch.testing.assert_close(za,aa,rtol=0,atol=0);self.assertGreater(float((zb-bb).abs().max()),1e-6)
    def test_checkpoint_recompute_gradients_and_unlabeled_exclusion(self):
        a=ChangeRelation(head(),activation_checkpoint=False).eval();b=copy.deepcopy(a);b.activation_checkpoint=True
        fs=organize([fixture(),dict(fixture(),pts=.2,frame_index=1)],'B_FULL_XY');fs[1]['supervised']=False
        ga=[]
        for m in [a,b]:
            z,aux=m(fs,'B_FULL_XY');loss,parts=video_loss(z,aux,fs);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.visual[0].weight.grad.abs().sum()),0)
            ga.append({n:p.grad.clone() if p.grad is not None else None for n,p in m.named_parameters()})
        for n in ga[0]:
            if ga[0][n] is not None:torch.testing.assert_close(ga[0][n],ga[1][n],rtol=1e-4,atol=1e-6)
    def test_restore_and_report_cpu_interface(self):
        m=torch.nn.Linear(2,1);opt=torch.optim.AdamW(m.parameters());m(torch.ones(1,2)).sum().backward();opt.step()
        rec=[dict(supervised_frames=1)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'last.pt';save_checkpoint(path,m,opt,{},'A_CURRENT_XY',1,1,rec);h=state_hash(m.state_dict())
            with torch.no_grad():m.weight.add_(2)
            restore_checkpoint(path,m,opt,{},'A_CURRENT_XY');self.assertEqual(h,state_hash(m.state_dict()))
            report(root);self.assertEqual(__import__('json').loads((root/'summary.json').read_text())['status'],'PARTIAL')
if __name__=='__main__':unittest.main()
