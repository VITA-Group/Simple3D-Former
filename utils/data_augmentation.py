# incorporate rotation and translation

from binvox_rw import Voxels, read_as_coord_array, read_as_3d_array, write
import numpy as np
from scipy.spatial.transform import Rotation as R

def add_affine_transformation_to_voxel(fp, fix_coords=True, test = False):
    """
    Add pertubation from 3D coordinates of voxel and then translate it back 
    Right now this is very rough and shall not be considered as the ultimate solution
    """

    # Get coords 
    vox = read_as_coord_array(fp, fix_coords=fix_coords)
    vox.data = (vox.data + 0.5) / np.array(vox.dims)[:,None]
    vox.data = vox.scale * vox.data + np.array(vox.translate)[:,None]

    translation = np.random.uniform(-0.1,0.1,3) * vox.scale
    omega = np.pi * np.random.uniform(0,0.2)
    rotvec = np.random.rand(3)
    rotvec = rotvec/ np.linalg.norm(rotvec)
    rotation = R.from_rotvec(omega * rotvec).as_matrix()
    scaling = np.random.uniform(0.9,1.1)
    #translation = np.zeros((3,))
    #rotation = np.eye(3)
    #scaling = 1.0
    if test:
        print(vox.data[:,1])
    new_coords = scaling * rotation @ vox.data + translation[:, None] 
    new_coords2 = scaling * rotation @ (vox.data + np.random.uniform(-0.01,0.01,3)[:, None] * vox.scale) + translation[:, None] 
    new_coords3 = scaling * rotation @ (vox.data + np.random.uniform(-0.01,0.01,3)[:, None] * vox.scale) + translation[:, None] 
    #new_coords4 = scaling * rotation @ (vox.data + np.random.uniform(-0.01,0.01,3)[:, None] * vox.scale) + translation[:, None] 
    if test:
        print(new_coords[:,1])

    new_coords = np.hstack((new_coords,new_coords2,new_coords3))
    if test:
        print(new_coords.shape)

    # Convert back in i,j,k
    convert = (new_coords - np.array(vox.translate)[:,None])/vox.scale
    if test:
        print(convert[:,1])
    indices = (convert * np.array(vox.dims)[:, None] - 0.5)
    min_idx = np.min(indices[:])
    if min_idx < 0:
        if test:
            print(min_idx)
        indices = indices - min_idx
    max_idx = np.max(indices[:])
    if max_idx >= vox.dims[0]:
        if test:
            print(max_idx)
        indices = indices + vox.dims[0] - max_idx    
    if test:
        print(indices[:,1].astype(float))
    new_vox = np.zeros(vox.dims, dtype=int)
    for i in range(indices.shape[1]):
        idx = (indices[:,i]).astype(int)
        if idx[0]<0 or idx[0]>=vox.dims[0]:
            continue
        if idx[1]<0 or idx[1]>=vox.dims[1]:
            continue
        if idx[2]<0 or idx[2]>=vox.dims[2]:
            continue
        new_vox[idx[0]][idx[1]][idx[2]] = 1
        #if idx[0]<vox.dims[0]-1 and idx[1]<vox.dims[1]-1 and idx[2]<vox.dims[2]-1:
        #    new_vox[idx[0]+1][idx[1]][idx[2]] = 1
        #    new_vox[idx[0]][idx[1]+1][idx[2]] = 1
        #    new_vox[idx[0]][idx[1]][idx[2]+1] = 1
        #    new_vox[idx[0]+1][idx[1]+1][idx[2]] = 1
        #    new_vox[idx[0]+1][idx[1]][idx[2]+1] = 1
        #    new_vox[idx[0]][idx[1]+1][idx[2]+1] = 1
        #    new_vox[idx[0]+1][idx[1]+1][idx[2]+1] = 1
    if test:
        fp.seek(0, 0)
        test_vox = read_as_3d_array(fp, fix_coords=fix_coords)
        print(np.count_nonzero(test_vox.data))
        print(np.count_nonzero(new_vox))
    vox.data = new_vox
    return vox

if __name__=="__main__":
    with open('./data/ModelNet10/bathtub/train/bathtub_0001.binvox','rb') as f:
        vox = add_affine_transformation_to_voxel(f, test=True)
    #with open('./test.binvox', 'wb') as f:
    #    write(vox, f)