"""Reuse full V8 attention/GRU/chunk path; only prepared coordinates differ."""
from torch import nn
from research_tools.v8.component_global_context_pilot.model import GlobalContextModel

CONDITIONS=('STRUCTURE_2D','STRUCTURE_3D')
# Both conditions: 67 RGB, 3 position, position-valid, track coverage,
# 3 matched-point displacement, displacement-valid, history, missing, 2 current UV.
INPUT_FIELDS=('ResNet18_layer1[64]','mean_RGB[3]','position[3]',
              'position_valid','track_coverage','matched_displacement[3]',
              'displacement_valid','history_available','position_missing','current_UV[2]')

class StructureModel(GlobalContextModel):
    def __init__(self,statistics):
        super().__init__('GLOBAL_COMPONENT',statistics,patch_chunk=256)
        self.cell_encoder=nn.Sequential(nn.Linear(80+64,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.input_dim=144
