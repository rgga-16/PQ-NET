from collections import OrderedDict
from tqdm import tqdm
from dataset import get_dataloader
from config import get_config
from util.utils import cycle
from agent import get_agent
import numpy as np

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
    # voxels = np.transpose(voxels, (0,1,3,2))
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
    # create experiment config
    config = get_config('pqnet')('train')

    # create network and training agent
    tr_agent = get_agent(config)

    # load from checkpoint if provided
    if config.cont:
        tr_agent.load_ckpt(config.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    for e in range(clock.epoch, config.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):

            """
            bs=40
            data:
            vox3d (bs, 1, 64,64,64)
            points(bs,4096,3)
            values(bs,4096,1)
            n_parts(bs)
            part_idx(bs)
            path(bs)
            """

            vox = data['vox3d'].squeeze().numpy()
            show_voxels(vox,n_rows=3)



            # train step
            outputs, losses = tr_agent.train_func(data)

            # visualize
            if config.vis and clock.step % config.vis_frequency == 0:
                tr_agent.visualize_batch(data, 'train', outputs=outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % config.val_frequency == 0:
                data = next(val_loader)
                outputs, losses = tr_agent.val_func(data)

                if config.vis and clock.step % config.vis_frequency == 0:
                    tr_agent.visualize_batch(data, 'validation', outputs=outputs)

            clock.tick()

        # update lr by scheduler
        tr_agent.update_learning_rate()

        # update teacher forcing ratio
        if config.module == 'seq2seq':
            tr_agent.update_teacher_forcing_ratio()

        clock.tock()
        if clock.epoch % config.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
