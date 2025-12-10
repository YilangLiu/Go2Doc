"""
Script to visualize saved pointcloud numpy files offline
Usage: python visualize_saved_pointcloud.py <path_to_npz_file>
"""

import numpy as np
import open3d as o3d
import sys
import os

def visualize_pointcloud(npz_path):
    """Load and visualize pointcloud from numpy file"""
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found")
        return
    
    # Load the numpy file
    data = np.load(npz_path)
    points = data['points']
    colors = data['colors']
    
    print(f"Loaded pointcloud with {len(points)} points")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape}")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    print("Visualizing pointcloud... (Press 'Q' or close window to exit)")
    o3d.visualization.draw_geometries([pcd], window_name="Saved Point Cloud")

def visualize_all_pointclouds(directory="pointcloud_data"):
    """Visualize all pointcloud files in a directory"""
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found")
        return
    
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        print(f"No .npz files found in {directory}")
        return
    
    print(f"Found {len(npz_files)} pointcloud files")
    
    # Load all pointclouds
    pcds = []
    for npz_file in sorted(npz_files):
        filepath = os.path.join(directory, npz_file)
        data = np.load(filepath)
        points = data['points']
        colors = data['colors']
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(pcd)
        print(f"Loaded {npz_file}: {len(points)} points")
    
    # Visualize all together
    print("Visualizing all pointclouds... (Press 'Q' or close window to exit)")
    o3d.visualization.draw_geometries(pcds, window_name="All Saved Point Clouds")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Visualize specific file
        visualize_pointcloud(sys.argv[1])
    else:
        # Visualize all files in default directory
        print("No file specified. Visualizing all pointclouds in 'pointcloud_data' directory...")
        visualize_all_pointclouds()


