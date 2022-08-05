import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import h5py
from dataset import get_dataloader
from config import get_config
from agent import PQNET
from util.utils import ensure_dir
from util.utils import cycle
from util.visualization import partsdf2voxel
from agent import get_agent
from networks import get_network
import numpy as np

from outside_code.libmise import MISE

import os
os.environ["QT_API"] = "pyqt5"

import pyvista as pv
from pyvistaqt import BackgroundPlotter, MultiPlotter


class Coarse2Parts(object):
    def __init__(self, config):
        self.config=config
        self.whole_encoder = None
        self.pqnet = PQNET(config)
        self.load_networks(config)
    
    def load_networks(self, config):
        """load trained network module: seq2seq and whole_ae"""
        self.pqnet.load_network(config)
        print("Loading PQNET")

        whole_imnet = get_network("whole_ae", config)
        whole_imnet.load_state_dict(torch.load(config.wholeae_modelpath)['model_state_dict'])
        print("Load WholeAE model from: {}".format(config.wholeae_modelpath))
        self.whole_encoder = whole_imnet.encoder.cuda().eval()
        self.whole_decoder = whole_imnet.decoder.cuda().eval()
    
    def infer_whole_encoder(self, coarse_voxel):
        """run part ae encoder to map coarse voxel to vectors
        :param coarse_voxel:  (1, 1, vox_dim, vox_dim, vox_dim)
        :return: coarse_code: (1, en_z_dim)
        """
        # whole_codes(1, en_z_dim)
        coarse_code = self.whole_encoder(coarse_voxel)   #WholeAE, encoder portion.
        return coarse_code
    
    '''
    WIP: Modify this code to convert the feature of encode whole shape into parts. 
    '''
    '''
    def decode_wholeshapefeat(config):
        """decode given latent codes to final shape"""
        # create the whole framwork
        pqnet = PQNET(config)

        # load source h5 file
        with h5py.File(config.fake_z_path, 'r') as fp:
            all_zs = fp['zs'][:] #(2000,1024)

        # output dest
        fake_name = config.fake_z_path.split('/')[-1].split('.')[0]
        save_dir = os.path.join(config.exp_dir, "results/{}-{}-p{}".format(fake_name, config.format,
                                                                                int(config.by_part)))
        ensure_dir(save_dir)

        # decoding
        pbar = tqdm(range(all_zs.shape[0]))
        for i in pbar:
            z = all_zs[i] #z(1024,)
            z1, z2 = np.split(z, 2) #z1(512,) , z2(512,)
            z = np.stack([z1, z2]) #z(2,512)
            z = torch.tensor(z, dtype=torch.float32).unsqueeze(1).cuda() #z(2,1,512)
            with torch.no_grad():
                pqnet.decode_seq(z)
                #output_shape(64,64,64) [0,1] or [0,n_parts] if by_part=True
                output_shape = pqnet.generate_shape(format=config.format, by_part=config.by_part) 

            data_id = "%04d" % i
            save_output(output_shape, data_id, save_dir, format=config.format)
    '''
    def c2p(self, coarse_voxel):

        #WIP: coarse_feat(bs,128) maybe this should have been (bs,1024)
        coarse_feat = self.infer_whole_encoder(coarse_voxel).squeeze()
        z1, z2 = np.split(coarse_feat.cpu().detach().numpy(), 2) #z1(512,) , z2(512,)
        z = np.stack([z1, z2]) #z(2,512)
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(1).cuda() #z(2,1,512)

        with torch.no_grad():
            self.pqnet.decode_seq(z)
            #output_shape(64,64,64) [0,1] or [0,n_parts] if by_part=True
            output_shape = self.pqnet.generate_shape(format=self.config.format, by_part=self.config.by_part) 
            print()
        return output_shape 

    
def main():
    # create experiment config
    config = get_config('pqnet')('test')
    coarse2parts = Coarse2Parts(config)

    vox_dim=64; upsampling_steps=0; threshold=0.5
    mesh_extractor = MISE(vox_dim, upsampling_steps, threshold)

    #Insert data loading here
    # create dataloader
    config.module='whole_ae'
    test_loader = get_dataloader('test', config)

    # begin iteration
    pbar = tqdm(test_loader)
    for b, data in enumerate(pbar):
        vox = data['vox3d'].cuda()
        parts = coarse2parts.c2p(vox)

        # points = mesh_extractor.query()
        # while points.shape[0] != 0:
        #     # Query points
        #     pointsf = torch.FloatTensor(points).cuda()
        #     # rescale to range (0, 1)
        #     pointsf = pointsf / mesh_extractor.resolution

        #     values = tr_agent.demo(vox, pointsf.unsqueeze(0)) #values(bs, 4,16384,1) if vox_dim==64
        #     values=values.squeeze()
        #     values_np = values.detach().cpu().numpy().astype(np.double)

        #     mesh_extractor.update(points, values_np)
        #     points = mesh_extractor.query()

        # all_points, all_values = mesh_extractor.get_points()

        # shape_voxel = partsdf2voxel([all_points],[all_values],vox_dim=64,by_part=False)

    # feed coarse vox and points here.

    return 
    
if __name__=='__main__':
    main()