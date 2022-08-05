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
import numpy as np

from outside_code.libmise import MISE

import os
os.environ["QT_API"] = "pyqt5"

import pyvista as pv
from pyvistaqt import BackgroundPlotter, MultiPlotter



def show_voxels(voxels,true_threshold=0.5,title='',n_rows=1):
    '''
    Args:
    voxel (numpy.ndarray(l,w,h)): A voxel grid of dimension l x w x h
    '''
    n_voxels = voxels.shape[0]
    n_cols = int(np.ceil(n_voxels/n_rows))
    voxels = np.transpose(voxels, (0,3,2,1))
    plotter = pv.Plotter(shape=(n_rows,n_cols),title=title)
    r = 0; c = 0
    for i in range(n_voxels):
        voxel = voxels[i]
        grid = pv.UniformGrid()
        grid.dimensions = voxel.shape
        grid.spacing = (1, 1, 1)  
        grid.point_data["values"] = voxel.flatten(order="F")  # Flatten the array!
        grid = grid.threshold(true_threshold, invert=False)
        plotter.subplot(r,c)
        plotter.add_mesh(grid,name=f'{r}_{c}',show_edges=True)
        c+=1
        if c==n_cols:
            r+=1
            c=0
        if r==n_rows:
            break
    plotter.show()
    return 

def main():

    vox_dim=64; upsampling_steps=0; threshold=0.5
    mesh_extractor = MISE(vox_dim, upsampling_steps, threshold)
    # create experiment config
    config = get_config('pqnet')('test')

    # create network and training agent
    tr_agent = get_agent(config)

    tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    test_loader = get_dataloader('test', config)

    # start training
    clock = tr_agent.clock

    # begin iteration
    pbar = tqdm(test_loader)
    for b, data in enumerate(pbar):

        """
        Data format for AE
        bs=40
        data:
        vox3d (bs, 1, 64,64,64). This is a part.
        points(bs,4096,3)
        values(bs,4096,1)
        n_parts(bs)
        part_idx(bs)
        path(bs)
        """

        vox = data['vox3d']
        # whole_vox = np.clip(np.sum(vox,axis=0,keepdims=True),0,1)
        # show_voxels(np.concatenate([whole_vox,vox],axis=0),n_rows=1)

        # test step
        # values = tr_agent.demo(vox, points) #values(bs, 4,16384,1) if vox_dim==64
        # values = tr_agent.test_func(data) #values(bs, 4,16384,1) if vox_dim==64

        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points).cuda()
            # rescale to range (0, 1)
            pointsf = pointsf / mesh_extractor.resolution

            values = tr_agent.demo(vox, pointsf.unsqueeze(0)) #values(bs, 4,16384,1) if vox_dim==64
            values=values.squeeze()
            values_np = values.detach().cpu().numpy().astype(np.double)

            mesh_extractor.update(points, values_np)
            points = mesh_extractor.query()

        all_points, all_values = mesh_extractor.get_points()

        shape_voxel = partsdf2voxel([all_points],[all_values],vox_dim=64,by_part=False)
        # visualize
        
        show_voxels(np.expand_dims(shape_voxel,0))

        clock.tick()


















if __name__ == '__main__':
    main()