
import numpy as np
import pandas as pd
import json 

from pyntcloud import PyntCloud
import pickle, h5py

from tqdm import tqdm
from util.utils import ensure_dir

import os
os.environ["QT_API"] = "pyqt5"

import pyvista as pv
# from pyvistaqt import BackgroundPlotter, MultiPlotter

def show_voxels(voxels,true_threshold=0.5,title='',n_rows=1):
    '''
    Args:
    voxel (numpy.ndarray(bs,l,w,h)): A voxel grid of dimension l x w x h
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

# Inspecting a data sample from processed data. After sampling points

src_root = 'data/Chair_raw'

dest_root = 'data/Chair_whole'
vox_dim=64


files = os.listdir(src_root)

ensure_dir(dest_root)

for i in tqdm(range(len(files))):
    file = files[i]
    srcpath = os.path.join(src_root,file)
    with h5py.File(srcpath,'r') as data:
        '''
        dict key-vals
        parts_voxel_scaled64: (n_parts,64,64,64)
        scales: (n_parts,1)
        shape_voxel64: (64,64,64)
        size: (n_parts,3)
        translations: (n_parts, 3)
        '''
        items = list(data.items())
        parts_voxel_scaled = data['parts_voxel_scaled64'][:]
        n_parts = parts_voxel_scaled.shape[0]
        # parts_merged = np.clip(np.sum(parts,axis=0,keepdims=True),0,1)
        whole = data['shape_voxel64'][:]

        scales = data['scales'][:]
        parts_size = data['size'][:]
        translations = data['translations'][:]
    
    whole_01 = np.clip(whole,0,1).astype(bool)

    destpath = os.path.join(dest_root,file)
    with h5py.File(destpath, 'a') as fp:
        fp.create_dataset("shape_voxel{}_bool".format(vox_dim), shape=(vox_dim, vox_dim, vox_dim),
                          dtype=bool, data=whole_01, compression=9)
        fp.create_dataset("shape_voxel{}".format(vox_dim), shape=(vox_dim, vox_dim, vox_dim),
                          dtype=np.uint8, data=whole, compression=9)
        fp.create_dataset('parts_voxel_scaled{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                          dtype=np.bool, data=parts_voxel_scaled, compression=9)
        fp.create_dataset('scales'.format(vox_dim), shape=(n_parts, 1),
                          dtype=np.float, data=scales, compression=9)
        fp.create_dataset('translations'.format(vox_dim), shape=(n_parts, 3),
                          dtype=np.float, data=translations, compression=9)
        fp.create_dataset('size'.format(vox_dim), shape=(n_parts, 3),
                          dtype=np.int, data=parts_size, compression=9)
        fp.attrs['n_parts'] = n_parts
        fp.attrs['name'] = file.encode('utf-8')
