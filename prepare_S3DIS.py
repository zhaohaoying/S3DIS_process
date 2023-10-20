


import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree


import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling





def convert_pointcloud2ply(read_path_name, save_path_name):
    """
    Convert original dataset files to numpy array files (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """

    data_list = []

    # extract all the txt files in the path
    for f in glob.glob(os.path.join(read_path_name, '*.txt')):
   
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:
            class_name = 'clutter'
        
        # split by whitespace
        pointcloud = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pointcloud.shape[0], 1))*gt_class2label[class_name]
        data_list.append(np.concatenate([pointcloud, labels], axis=1))
    
    

    # aggregate all the points in the room
    full_point_features = np.concatenate(data_list, 0)

    xyz_min = np.amin(full_point_features, axis=0)[0:3]
    full_point_features[:, 0:3] -= xyz_min

    

    # save the full scene
    np.save(save_path_name + '_full_scene.npy', full_point_features)

    xyz = full_point_features[:, :3].astype(np.float32)
    colors = full_point_features[:, 3:6].astype(np.uint8)
    labels = full_point_features[:, 6].astype(np.uint8)



    # save the scene sampled by grid sampling method
    sub_xyz, sub_colors, sub_labels = cpp_subsampling.compute(xyz, features=colors, classes=labels, sampleDl=0.04, verbose=0)
    sub_colors = sub_colors / 255.0

    np.save(save_path_name + '_full_scene_grid.npy', np.concatenate((sub_xyz, sub_colors, sub_labels), axis=1))


    # save the KDTree

    search_tree = KDTree(sub_xyz)

    kd_tree_filename = save_path_name + '_KDTree.pkl'
    with open(kd_tree_filename, 'wb') as f:
        pickle.dump(search_tree, f)

    # save the index of the sampled points
    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))        
    proj_idx = proj_idx.astype(np.int32)
    
    index_filename = save_path_name + '_proj.npy'
    np.save(index_filename, proj_idx)
    



if __name__ == '__main__':
    dataset_path = '/home/zhy/Dataset/S3DIS_aligned/Stanford3dDataset_v1.2_Aligned_Version' 
    output_path = os.path.join('/home/zhy/Dataset/S3DIS_aligned', 'input_0.040')

    # Read the file name and combine with the dataset path
    annotation_files_path = [line.rstrip() for line in open('anno_paths.txt')]
    full_file_path = [os.path.join(dataset_path, p) for p in annotation_files_path]



    # Generate a class name dictionary
    gt_class = [x.rstrip() for x in open('class_names.txt')]
    gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

    for filename in annotation_files_path:
        print(filename)
        # eg. Area_1/office_12
        elements = str(filename).split('/')
        
        # naming the files based on the area and room names
        out_file_name = elements[-3] + '_' + elements[-2]

        convert_pointcloud2ply(os.path.join(dataset_path, filename), os.path.join(output_path, out_file_name))












