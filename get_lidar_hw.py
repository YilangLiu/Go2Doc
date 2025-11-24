from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.sensor_msgs.msg.dds_._PointCloud2_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_._HeightMap_ import HeightMap_
import numpy as np
import sys
import time
from collections import deque
import open3d as o3d

class Custom:
    def __init__(self):
        self.pointcloud_topic = "rt/utlidar/cloud"
        self.heightmap_topic = "rt/utlidar/height_map_array"
        self.heightmap_memory_size = 500
        self.point_cloud_msg = None
        self.point_cloud_updated = False
        self.vis = None
        self.pcd = None
        self.running = True
    
    def Init(self):
        self.point_cloud_subscriber = ChannelSubscriber(self.pointcloud_topic, PointCloud2_)
        self.height_map_subscriber = ChannelSubscriber(self.heightmap_topic, HeightMap_)
        self.point_cloud_subscriber.Init(self.PointCloud_Handler, 10)
        self.height_map_subscriber.Init(self.HeightMap_Handler, 10)
        self.point_cloud_data = np.zeros(10)
        self.height_map_msg_debug = None
        self.height_map_data = deque(maxlen=self.heightmap_memory_size)
        
        # Initialize Open3D visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="LiDAR Point Cloud")
        self.pcd = o3d.geometry.PointCloud()
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        self.vis.add_geometry(frame)
        
    def PointCloud_Handler(self, msg: PointCloud2_):
        # print("Receive point cloud data! \n")
        # print(f"stamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec} \n")
        # print(f"frame = {msg.header.frame_id} \n")
        # print(f"point cloud width {msg.width}, height: {msg.height} \n")
        self.point_cloud_msg = msg
        self.point_cloud_updated = True

    def HeightMap_Handler(self, msg: HeightMap_):
        # print("Receive Height Map data! \n")
        # print(f"stamp {msg.frame_id} \n")
        # print(f"frame = {msg.header.frame_id} \n")
        # print(f"point cloud width {msg.width}, height: {msg.height} \n")
        # print(f"point cloud resolution msg.resolution \n")

        self.height_map_msg_debug = msg

        width = msg.width
        height = msg.height
        resolution = msg.resolution
        originX = msg.origin[0]
        originY = msg.origin[1]
        
        height_point = np.zeros(3)
        for iy in range(height):
            for ix in range(width):
                index = ix + width * iy
                height_point[2] = msg.data[index]
                if height_point > 1e9:
                    continue
                height_point[0] = ix * resolution + originX
                height_point[1] = iy * resolution + originY
                self.height_map_data.append(height_point)
        
    def parse_pointcloud(self, msg: PointCloud2_):
        """Parse PointCloud2_ message to extract xyz coordinates"""
        if msg.width == 0 or len(msg.data) == 0:
            return None
        
        # Find x, y, z field offsets
        x_offset = None
        y_offset = None
        z_offset = None
        intensity_offset = None
        
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
            elif field.name == 'intensity':
                intensity_offset = field.offset
        
        if x_offset is None or y_offset is None or z_offset is None:
            print("Warning: Could not find x, y, z fields in pointcloud")
            return None
        
        # Parse points
        num_points = msg.width * msg.height
        points = np.zeros((num_points, 3))
        colors = np.ones((num_points, 3)) * 0.5  # Default gray color
        
        # Convert cyclonedds sequence to bytes/numpy array
        # Handle both list and sequence types
        if isinstance(msg.data, list):
            data_bytes = bytes(msg.data)
        else:
            data_bytes = bytes(msg.data) if hasattr(msg.data, '__iter__') else msg.data
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)
        
        for i in range(num_points):
            point_start = i * msg.point_step
            if point_start + 4 > len(data_array):
                break
            
            # Extract x, y, z (assuming float32)
            x_bytes = bytes(data_array[point_start + x_offset:point_start + x_offset + 4])
            y_bytes = bytes(data_array[point_start + y_offset:point_start + y_offset + 4])
            z_bytes = bytes(data_array[point_start + z_offset:point_start + z_offset + 4])
            
            # Handle endianness
            dtype = '>f4' if msg.is_bigendian else '<f4'
            x = np.frombuffer(x_bytes, dtype=dtype)[0]
            y = np.frombuffer(y_bytes, dtype=dtype)[0]
            z = np.frombuffer(z_bytes, dtype=dtype)[0]
            
            points[i] = [x, y, z]
            
            # Extract intensity if available (for coloring)
            if intensity_offset is not None:
                intensity_bytes = bytes(data_array[point_start + intensity_offset:point_start + intensity_offset + 4])
                intensity = np.frombuffer(intensity_bytes, dtype=dtype)[0]
                # Normalize intensity to color (0-1 range)
                intensity_norm = min(intensity / 255.0, 1.0) if intensity > 0 else 0.0
                colors[i] = [intensity_norm, intensity_norm, intensity_norm]
        
        # Remove invalid points (NaN or inf)
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        return points, colors
    
    def update_visualization(self):
        """Update Open3D visualization with new pointcloud data"""
        if self.point_cloud_msg is None or not self.point_cloud_updated:
            return
        
        self.point_cloud_updated = False
        
        try:
            result = self.parse_pointcloud(self.point_cloud_msg)
            if result is None:
                return
            
            points, colors = result
            
            if len(points) == 0:
                return
            
            # Update point cloud
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Add geometry if not already added
            if len(self.vis.get_geometry_list()) == 1:  # Only coordinate frame
                self.vis.add_geometry(self.pcd)
            else:
                self.vis.update_geometry(self.pcd)
            
            self.vis.poll_events()
            self.vis.update_renderer()
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
    
    def Start(self):
        try:
            while True:
                # Update visualization in main thread
                self.update_visualization()
                
                try:
                    print(f"receive {self.point_cloud_msg.header.frame_id} with number: {self.point_cloud_msg.width}")
                except:
                    print("no point cloud data received yet.")
                
                try:
                    print(f"receive {self.height_map_msg_debug.frame_id} {self.height_map_data[0]}")
                except:
                    print("no height map data received yet.")
                time.sleep(0.033)  # ~30 fps for visualization
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
            if self.vis:
                self.vis.destroy_window()

if __name__ == "__main__":
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()
