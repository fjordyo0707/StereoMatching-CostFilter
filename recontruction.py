import numpy as np
import cv2
from open3d import *
import numpy.matlib

def main():
    scale_factor = 1
    img_left = cv2.imread('./oil_venus.png')  
    #img_right = cv2.imread('./testdata/tsukuba/im4.png')    
    img_disparity = cv2.imread('./venus.png', cv2.IMREAD_GRAYSCALE)/scale_factor
    
    h, w = img_disparity.shape

    xyz_points = np.zeros((h*w, 6))

    Y = np.matlib.repmat(np.array(range(h)).reshape(-1,1), 1, w).flatten().astype(int)
    X = np.matlib.repmat(np.array(range(w)), h,1 ).flatten().astype(int)
    Z = img_disparity.flatten().astype(int)

    B = img_left[:,:,0].flatten()/255
    G = img_left[:,:,1].flatten()/255
    R = img_left[:,:,2].flatten()/255
    
    xyz_points[:,0] = X
    xyz_points[:,1] = Y
    xyz_points[:,2] = Z
    xyz_points[:,3] = R
    xyz_points[:,4] = G
    xyz_points[:,5] = B
    
    
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz_points[:,:3])
    pcd.colors = Vector3dVector(xyz_points[:,3:])
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    draw_geometries([pcd])
    write_point_cloud("./pointcloud/venus.ply", pcd)
    




if __name__ == '__main__':
    main()