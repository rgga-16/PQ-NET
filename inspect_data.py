import numpy as np
import pandas as pd
import json 

from pyntcloud import PyntCloud
import pickle, h5py

from tqdm import tqdm

from dataset.data_utils import collect_data_id, load_from_hdf5_by_part

import os
os.environ["QT_API"] = "pyqt5"

import pyvista as pv
# from pyvistaqt import BackgroundPlotter, MultiPlotter

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

# Inspecting a data sample from raw data. Done after run_postprocess.sh script in Data Prep Stage.
# Before sampling points
datapath = 'data\\Chair_raw\\172.h5'
with h5py.File(datapath,'r') as data:
    '''
    dict key-vals
    parts_voxel_scaled64: (n_parts,64,64,64)
    scales: (n_parts,1)
    shape_voxel64: (64,64,64)
    size: (n_parts,3)
    translations: (n_parts, 3)
    '''
    parts = data['parts_voxel_scaled64'][:].astype(np.float)
    parts_merged = np.clip(np.sum(parts,axis=0,keepdims=True),0,1)
    whole = np.expand_dims(data['shape_voxel64'][:].astype(np.float),0)
    scales = data['scales'][:] #ex. array([[0.515625]])
    size = data['size'][:] #ex. array([[33,26,10],...])
    translations = data['translations'][:] #ex. array([[43.,31.5,24.5],...]). translations is just midpoint.
    show_voxels(np.concatenate([whole,parts_merged,parts],axis=0))
    print()

# Inspecting a data sample from processed data. After sampling points
datapath = 'data\\Chair\\172.h5'
with h5py.File(datapath,'r') as data:
    '''
    dict key-vals
    parts_voxel_scaled64: (n_parts,64,64,64)
    scales: (n_parts,1)
    shape_voxel64: (64,64,64)
    size: (n_parts,3)
    translations: (n_parts, 3)
    points16: (n_parts, 4096,3)
    points32: (n_parts, 8192,3)
    points64: (n_parts, 32768, 3)
    values16: (n_parts, 4096,1)
    values32: (n_parts, 8192,1)
    values64: (n_parts, 32768,1)
    '''
    items = list(data.items())
    parts = data['parts_voxel_scaled64'][:].astype(np.float)
    parts_merged = np.clip(np.sum(parts,axis=0,keepdims=True),0,1)
    whole = np.expand_dims(data['shape_voxel64'][:].astype(np.float),0)
    scales = data['scales'][:]
    size = data['size'][:]
    translations = data['translations'][:]

    points16 = data['points_16'][:]
    points32 = data['points_32'][:]
    points64  = data['points_64'][:]

    values16 = data['values_16'][:]
    values32 = data['values_32'][:]
    values64 = data['values_64'][:]
    show_voxels(np.concatenate([whole,parts],axis=0))
    print()




shape_names = collect_data_id('Chair', 'train')
with open('data/{}_info.json'.format('Chair'), 'r') as fp:
    nparts_dict = json.load(fp)
parts_info = []
for name in shape_names:
    shape_h5_path = os.path.join('data\\Chair', name + '.h5')
    if not os.path.exists(shape_h5_path):  # check file existence
        continue
    nparts = nparts_dict[name]
    parts_info.extend([(shape_h5_path, x) for x in range(nparts)])

index = 1766