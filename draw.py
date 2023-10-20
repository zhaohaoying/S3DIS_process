import numpy as np
import open3d

def draw_pc(pc_xyzrgb):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
    open3d.visualization.draw_geometries([pc])



scene = np.load('/home/zhy/Dataset/S3DIS_aligned/input_0.040/Area_1_WC_1_full_scene_grid.npy')
draw_pc(scene)


