import torch
import torch.nn as nn
from networks import get_network
from agent.base import BaseAgent
from util.visualization import project_voxel_along_xyz, visualize_sdf

class SVRAgent(BaseAgent):
    def __init__(self, config):
        super(SVRAgent, self).__init__(config)
        self.points_batch_size = config.points_batch_size
        self.resolution = config.resolution
        self.batch_size = config.batch_size