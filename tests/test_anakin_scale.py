import unittest
from research_tools.rgbd_slice.anakin_scale import select_candidates

class SelectionTests(unittest.TestCase):
    def row(self,pair,role,kind='inpainting',split='train'):
        return dict(pair_id=pair,role=role,generator=kind,split=split,source_id=pair,original_url='https://vimeo.com/'+pair,video_sha256=pair+role)
    def test_exclusion_balance_and_determinism(self):
        b=[self.row('base'+str(i),role,kind) for i,kind in enumerate(['inpainting']*4+['splicing']*4) for role in ['real','fake']]
        rows=[self.row(str(i),role,kind) for i,kind in enumerate(['inpainting']*5+['splicing']*5) for role in ['real','fake']]
        rows+=[self.row('held',role,split='test_reserved') for role in ['real','fake']]
        rows+=b
        a,t,excluded=select_candidates(rows,b);aa,tt,_=select_candidates(list(reversed(rows)),b)
        self.assertEqual(t,{'inpainting':4,'splicing':4});self.assertEqual(a,aa)
        self.assertFalse(any(r['pair_id']=='held' for rr in a.values() for r in rr));self.assertEqual(len(excluded),9)
    def test_duplicate_source_and_media(self):
        b=[self.row('base'+str(i),role,kind) for i,kind in enumerate(['inpainting']*4+['splicing']*4) for role in ['real','fake']]
        rows=[self.row('x',role) for role in ['real','fake']]+[self.row('y',role) for role in ['real','fake']]
        for r in rows:
            if r['pair_id']=='y':r['original_url']='http://www.vimeo.com/x/'
        g,_,excluded=select_candidates(rows,b);self.assertEqual(len(g['inpainting']),1);self.assertEqual(excluded[0]['pair_id'],'y')
if __name__=='__main__':unittest.main()
