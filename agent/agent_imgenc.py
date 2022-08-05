import torch
import torch.nn as nn
from networks import get_network
from agent.base import BaseAgent
from util.visualization import project_voxel_along_xyz, visualize_sdf

class ImgEncAgent(BaseAgent):
    def __init__(self, config):
        super(ImgEncAgent, self).__init__(config)
        self.points_batch_size = config.points_batch_size
        self.resolution = config.resolution
        self.batch_size = config.batch_size
    
    def build_net(self, config):
        net = get_network('imgenc', config)
        print('-----ResNet Image Encoder architecture-----')
        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net
    
    def set_loss_function(self):
        self.criterion = nn.MSELoss().cuda()
    
    def forward(self, data):
        image = data['img'].cuda()
        input_vox3d = data['vox3d'].cuda()  # (shape_batch_size, 1, dim, dim, dim)
        points = data['points'].cuda()  # (shape_batch_size, points_batch_size, 3)
        target_sdf = data['values'].cuda()  # (shape_batch_size, points_batch_size, 1)

        output_sdf = self.net(points, input_vox3d)

        loss = self.criterion(output_sdf, target_sdf)
        return output_sdf, {"mse": loss}