import unittest
import torch
from research_tools.xyz_pilot.model import route
from research_tools.xyz_pilot.relation_review import routing_evidence


class RoutingReviewTest(unittest.TestCase):
    def test_actual_route_floor_xyz_and_members(self):
        xyz = torch.tensor([[[-.01, 0., .2], [.1, .1, .2], [.7, .1, .2], [9., 9., 9.]]])
        valid = torch.tensor([[True, True, True, False]])
        result = route(xyz, valid, .5)
        evidence = routing_evidence(xyz, valid, .5, *result)
        self.assertEqual(result[1].tolist(), [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
        self.assertEqual(evidence['valid_points'], 3)
        self.assertEqual(evidence['member_count_sum'], 3)
        self.assertEqual(len(evidence['same_z_distinct_xy']), 3)

    def test_corrupt_inverse_is_rejected(self):
        xyz = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
        valid = torch.ones((1, 2), dtype=torch.bool)
        points, keys, index = route(xyz, valid, .5)
        with self.assertRaises(AssertionError):
            routing_evidence(xyz, valid, .5, points, keys, index.flip(0))
