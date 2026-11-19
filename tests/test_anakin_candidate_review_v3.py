import copy
import unittest
from scripts.anakin_candidate_review_v3 import choose


def entry(id,role='candidate_unassigned',stable=True,state='较稳定候选'):
    return dict(metadata={'video-id':id},role=role,source_id='source:'+id,camera_auto={'stable_candidate':stable,'state':state})


class NextReviewTests(unittest.TestCase):
    def test_role_human_previous_and_movement_exclusion(self):
        s={'selected':['old'],'prescreen':[entry('old'),entry('human'),entry('dev','dev'),entry('test','test_reserved'),entry('moving',stable=False,state='可见全局运动迹象'),entry('stable'),entry('unknown',stable=False,state='不确定')]}
        original=copy.deepcopy(s);a,ex,_=choose(s,{'an:human'})
        self.assertEqual(s,original);self.assertEqual([p['metadata']['video-id'] for p in a],['stable','unknown']);self.assertEqual(len(ex),5)
        self.assertEqual(a,choose(s,{'an:human'})[0])

    def test_bound_and_stable_priority_no_quota_filling(self):
        s={'selected':[],'prescreen':[entry(str(i)) for i in range(20)]}
        a,_,remaining=choose(s,set());self.assertEqual(len(a),12);self.assertEqual(len(remaining),8)
        s={'selected':[],'prescreen':[entry('one'),entry('moving',stable=False,state='可见全局运动')]}
        self.assertEqual(len(choose(s,set())[0]),1)
