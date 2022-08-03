
import numpy as np
import pandas as pd
import json 

from pyntcloud import PyntCloud
import pickle, h5py

from tqdm import tqdm

from dataset.data_utils import collect_data_id, load_from_hdf5_by_part
from voxelization.rescale_part_vox import safe_minmax, find_bounding_box
from skimage.transform import resize

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

# # Inspecting a data sample from raw data. Done after run_postprocess.sh script in Data Prep Stage.
# # Before sampling points
# datapath = 'data/Chair/172.h5'
# with h5py.File(datapath,'r') as data:
#     '''
#     dict key-vals
#     parts_voxel_scaled64: (n_parts,64,64,64)
#     scales: (n_parts,1)
#     shape_voxel64: (64,64,64)
#     size: (n_parts,3)
#     translations: (n_parts, 3)
#     '''
#     parts = data['parts_voxel_scaled64'][:].astype(np.float)
#     parts_merged = np.clip(np.sum(parts,axis=0,keepdims=True),0,1)
#     whole = np.expand_dims(data['shape_voxel64'][:].astype(np.float),0)
#     scales = data['scales'][:] #ex. array([[0.515625]])
#     sizes = data['size'][:] #ex. array([[33,26,10],...]) #Sizes of original parts before they got scaled.
#     translations = data['translations'][:] #ex. array([[43.,31.5,24.5],...]). translations is just midpoint.

#     d=1
#     dim_voxel=64

#     voxel_unscaled64 = None

#     for part,scale,translation,size in zip(parts,scales,translations,sizes):
#         # Denormalize. global_shape = unit_scaled_shape * scale + translation ????
#         bbox = find_bounding_box(part)
#         mins = np.asarray(list(map(lambda x: max(x - d, 0), bbox[::2])))
#         maxs = np.asarray(list(map(lambda x: min(x + d, dim_voxel - 1), bbox[1::2])))
#         axis_lengths = maxs - mins + 1

#         inverse_scale = 1/scale
#         orig_axis_lengths = dim_voxel/inverse_scale

#         bbox_voxel = part[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]

#         bbox_voxel_unscaled = resize(bbox_voxel, size, mode='constant')
#         bbox_voxel_unscaled = np.asarray(bbox_voxel_unscaled >= 0.5, dtype=float)
#         x_len, y_len, z_len = bbox_voxel_unscaled.shape

#         center = dim_voxel // 2
#         new_mins = list((center - x_len // 2, center - y_len // 2, center - z_len // 2))
        
#         part_unscaled64 = np.zeros((dim_voxel, dim_voxel, dim_voxel), dtype=float)
#         part_unscaled64[new_mins[0]:new_mins[0] + x_len,
#                        new_mins[1]:new_mins[1] + y_len,
#                        new_mins[2]:new_mins[2] + z_len] = bbox_voxel_unscaled
        
#         if voxel_unscaled64 is None:
#             voxel_unscaled64 = part_unscaled64
#         else:
#             voxel_unscaled64 += part_unscaled64
    
#         print()

#     show_voxels(np.expand_dims(voxel_unscaled64,0))
#     show_voxels(np.concatenate([whole,parts_merged,parts],axis=0))
#     print()

# Inspecting a data sample from processed data. After sampling points
datapath = 'data\\Airplane\\1a04e3eab45ca15dd86060f189eb133.h5' #Windows
datapath = 'data/Airplane/4fd9c86e43a1dea17209009cfb89d4bd.h5' #LInux
datapath = 'data/Chair/2465.h5' #Linux
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
    show_voxels(np.concatenate([whole,parts_merged,parts],axis=0),n_rows=2)
    # show_voxels(whole)
    print()



# Inspecting a data sample from data after running fill_part_solid.
# datapath = 'data\\Airplane\\1a04e3eab45ca15dd86060f189eb133.h5' #Windows
# datapath = 'voxelization/refined/Airplane/4fd9c86e43a1dea17209009cfb89d4bd.h5' #LInux
# datapath = 'data/Lamp/17897.h5' #LInux
# with h5py.File(datapath,'r') as data:
#     '''
#     dict key-vals
#     parts_voxel_scaled64: (n_parts,64,64,64)
#     scales: (n_parts,1)
#     shape_voxel64: (64,64,64)
#     size: (n_parts,3)
#     translations: (n_parts, 3)
#     points16: (n_parts, 4096,3)
#     points32: (n_parts, 8192,3)
#     points64: (n_parts, 32768, 3)
#     values16: (n_parts, 4096,1)
#     values32: (n_parts, 8192,1)
#     values64: (n_parts, 32768,1)
#     '''
#     items = list(data.items())
#     parts = data['parts_voxel64'][:].astype(np.float)
#     parts_merged = np.clip(np.sum(parts,axis=0,keepdims=True),0,1)
#     whole = np.expand_dims(data['shape_voxel64'][:].astype(np.float),0)
#     # scales = data['scales'][:]
#     # size = data['size'][:]
#     # translations = data['translations'][:]
#     show_voxels(np.concatenate([whole,parts_merged,parts],axis=0),n_rows=2)
#     # show_voxels(whole)
#     print()

# Inspecting a data sample from processing data. After sampling points on both whole and parts of the voxel.
datapath = 'data/Chair_whole_points_toy/182.h5' #Linux
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
    whole = np.expand_dims(data['shape_voxel64'][:].astype(np.float),0)
    whole_bool = np.expand_dims(data['shape_voxel64_bool'][:].astype(np.float),0)
    scales = data['scales'][:]
    size = data['size'][:]
    translations = data['translations'][:]

    points16 = data['wholepoints_16'][:]
    points32 = data['wholepoints_32'][:]
    points64  = data['wholepoints_64'][:]

    values16 = data['wholevalues_16'][:]
    values32 = data['wholevalues_32'][:]
    values64 = data['wholevalues_64'][:]
    show_voxels(np.concatenate([whole,parts],axis=0),n_rows=2)
    # show_voxels(whole)
    print()


# shape_names = collect_data_id('Chair', 'train')
# with open('data/{}_info.json'.format('Chair'), 'r') as fp:
#     nparts_dict = json.load(fp)
# parts_info = []
# for name in shape_names:
#     shape_h5_path = os.path.join('data\\Chair', name + '.h5')
#     if not os.path.exists(shape_h5_path):  # check file existence
#         continue
#     nparts = nparts_dict[name]
#     parts_info.extend([(shape_h5_path, x) for x in range(nparts)])

# index = 1766