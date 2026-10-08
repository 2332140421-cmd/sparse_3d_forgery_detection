from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research_tools.v8.local_geometry_pilot.data import cell_mean, components, patches
from research_tools.v8.local_geometry_pilot.model import LocalGeometryModel, PatchEncoder, geometry_input
from research_tools.v8.local_geometry_pilot.runner import validate_checkpoint

STATS = {"absolute_mean": [0., 0., 0.], "absolute_scale": [1., 1., 1.], "relative_scale": [1., 1., 1.]}
torch.set_num_threads(2)


def batch():
    torch.manual_seed(8)
    content = torch.zeros((1, 2, 256, 16, 16), dtype=torch.bool)
    content[:, :, :3] = True
    content[:, :, 2, :, 8:] = False
    valid = content.clone(); valid[:, :, 1] = False
    xyz = torch.randn(1, 2, 256, 16, 16, 3)
    mu = (xyz * valid[..., None]).sum((3, 4)) / valid.sum((3, 4)).clamp_min(1)[..., None]
    area = content.sum((3, 4)).float()
    return {"x": torch.randn(1, 2, 256, 77), "xyz": xyz, "valid": valid, "content": content, "mu": mu,
            "area": area, "component_ids": torch.arange(256)[None, None].expand(1, 2, 256).clone(),
            "predecessor": torch.zeros((1, 2, 256, 4), dtype=torch.long),
            "association_weight": torch.zeros(1, 2, 256, 4), "history_available": area.bool(), "delta_t": torch.tensor([[0., .1]])}


def test_mean_control_and_same_mean_different_shape():
    b = batch(); absmean, absscale, relscale = [torch.tensor(STATS[k]) for k in STATS]
    dense = geometry_input(b["xyz"], b["valid"], b["content"], b["mu"], absmean, absscale, relscale, "DENSE_LOCAL")
    mean = geometry_input(b["xyz"], b["valid"], b["content"], b["mu"], absmean, absscale, relscale, "MEAN_REPLICATED")
    assert torch.count_nonzero(mean[:, 3:6]) == 0
    torch.testing.assert_close(mean[:, 6], dense[:, 6])
    first = mean[0, :3].permute(1, 2, 0)
    torch.testing.assert_close(first, b["mu"][0, 0, 0].expand_as(first))
    changed = copy.deepcopy(b)
    changed["xyz"][:, :, 0] = torch.flip(changed["xyz"][:, :, 0], (2,))
    mean2 = geometry_input(changed["xyz"], changed["valid"], changed["content"], changed["mu"], absmean, absscale, relscale, "MEAN_REPLICATED")
    torch.testing.assert_close(mean, mean2, atol=0, rtol=0)
    dense2 = geometry_input(changed["xyz"], changed["valid"], changed["content"], changed["mu"], absmean, absscale, relscale, "DENSE_LOCAL")
    assert not torch.equal(dense, dense2)


def test_padding_geometry_missing_no_history_and_strict_exclusion():
    b = batch(); model = LocalGeometryModel("DENSE_LOCAL", STATS, patch_chunk=512).eval()
    altered = copy.deepcopy(b); active = b["area"] > 0
    altered["x"][~active] = 1e6
    altered["xyz"][~b["content"]] = 1e6
    altered["valid"][~b["content"]] = True
    altered["mu"][~active] = 1e6
    # Nonzero candidates that point only to padding must reduce to no-history.
    altered["predecessor"][:, 1, :3] = 250
    altered["association_weight"][:, 1, :3] = .25
    with torch.no_grad(): a, c = model(b, True), model(altered, True)
    torch.testing.assert_close(a["window_logit"], c["window_logit"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a["cell_logits"][active], c["cell_logits"][active], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(a["window_logit"]).all() and torch.isfinite(a["cell_logits"][active]).all()
    assert torch.count_nonzero(a["geometry_embedding"][~active]) == 0


def test_patch_chunk_gradient_and_save_reload(tmp_path):
    b = batch(); model = LocalGeometryModel("DENSE_LOCAL", STATS, 512)
    other = copy.deepcopy(model); other.patch_chunk = 73
    za, zb = model(b), other(b)
    torch.testing.assert_close(za, zb, atol=1e-6, rtol=1e-6)
    za.sum().backward(); zb.sum().backward()
    for a, c in zip(model.parameters(), other.parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        torch.testing.assert_close(a.grad, c.grad, atol=2e-6, rtol=2e-4)
    assert model.patch_encoder.conv1.weight.grad.abs().sum() > 0
    assert model.gru.weight_ih.grad.abs().sum() > 0
    path = tmp_path / "model.pt"; torch.save(model.state_dict(), path)
    loaded = LocalGeometryModel("DENSE_LOCAL", STATS, 512); loaded.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model(b), loaded(b), atol=0, rtol=0)
    mean = LocalGeometryModel("MEAN_REPLICATED", STATS)
    assert sum(p.numel() for p in mean.parameters()) == sum(p.numel() for p in model.parameters())


def test_conv_responds_to_spatial_arrangement():
    torch.manual_seed(54); encoder = PatchEncoder()
    x = torch.rand(1, 7, 16, 16); content = torch.ones(1, 16, 16, dtype=torch.bool)
    shuffled = x.flatten(2)[:, :, torch.randperm(256)].reshape_as(x)
    with torch.no_grad(): a, b = encoder(x, content), encoder(shuffled, content)
    assert (a - b).abs().max() > 1e-4


def test_content_mean_and_padding_never_bridges_groups():
    image = np.zeros((16, 256, 256, 3), np.float32)
    image[:, :16, :16] = 3.; image[:, :16, 16:32] = 100.
    content = np.zeros((16, 256, 256), bool); content[:, :16, :8] = True
    mu, gv, valid = cell_mean(patches(image), patches(np.ones_like(content)), patches(content))
    np.testing.assert_array_equal(mu[:, 0], 3.)
    assert not gv[:, 1].any()
    active = np.zeros(256, bool); active[[0, 2]] = True
    comp = components(np.zeros((256, 3)), np.ones(256, bool), np.zeros((256, 2)), np.ones(256, bool), active,
                      {"spatial_scale": 1., "motion_scale": 1., "merge_threshold": 1.})
    assert comp[0] != comp[2] and comp[0] != comp[1]


def test_effective_batch_weighted_gradient_equivalence():
    b = batch()
    b = {k: torch.cat((v, v), 0) for k, v in b.items()}
    full = LocalGeometryModel("DENSE_LOCAL", STATS, 512)
    micro = copy.deepcopy(full)
    targets = torch.tensor([0., 1.]); weights = torch.tensor([.3, 1.7])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(full(b), targets, weight=weights)
    loss.backward()
    for i in range(2):
        one = {k: v[i:i + 1] for k, v in b.items()}
        (torch.nn.functional.binary_cross_entropy_with_logits(micro(one), targets[i:i + 1], weight=weights[i:i + 1], reduction="sum") / 2).backward()
    for a, c in zip(full.parameters(), micro.parameters()):
        torch.testing.assert_close(a.grad, c.grad, atol=2e-6, rtol=3e-4)


def test_checkpoint_identity_and_epoch_rng_continuity():
    identity = {"input_id": "valid", "config_id": "cfg", "condition": "DENSE_LOCAL", "seed": 3}
    rng = torch.Generator().manual_seed(45); torch.randperm(7, generator=rng)
    saved = rng.get_state(); expected = torch.randperm(7, generator=rng)
    restored = torch.Generator(); restored.set_state(saved)
    torch.testing.assert_close(expected, torch.randperm(7, generator=restored), atol=0, rtol=0)
    payload = {"identity": identity, "model": {}, "optimizer": {}, "epoch": 1, "loss_history": [.5], "rng": {},
               "order_rng": saved, "training_elapsed_s": 1., "microbatch": 2, "peak_allocated_gib": 2., "initial_parameter_sha256": "i"}
    validate_checkpoint(payload, identity)
    for key, value in (("input_id", "wrong"), ("condition", "MEAN_REPLICATED"), ("config_id", "new")):
        with pytest.raises(ValueError): validate_checkpoint(payload, {**identity, key: value})
