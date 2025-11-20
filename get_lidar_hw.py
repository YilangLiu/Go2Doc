from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.sensor_msgs.msg.dds_._PointCloud2_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_._HeightMap_ import HeightMap_
import numpy as np
import sys
import time
from collections import deque

class Custom:
    def __init__(self):
        self.pointcloud_topic = "rt/utlidar/cloud"
        self.heightmap_topic = "rt/utlidar/height_map_array"
        self.heightmap_memory_size = 500
    
    def Init(self):
        self.point_cloud_subscriber = ChannelSubscriber(self.pointcloud_topic, PointCloud2_)
        self.height_map_subscriber = ChannelSubscriber(self.heightmap_topic, HeightMap_)
        self.point_cloud_subscriber.Init(self.PointCloud_Handler, 10)
        self.height_map_subscriber.Init(self.HeightMap_Handler, 10)
        self.point_cloud_data = np.zeros(10)
        self.height_map_msg_debug = None
        self.height_map_data = deque(maxlen=self.heightmap_memory_size)
        
    def PointCloud_Handler(self, msg: PointCloud2_):
        # print("Receive point cloud data! \n")
        # print(f"stamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec} \n")
        # print(f"frame = {msg.header.frame_id} \n")
        # print(f"point cloud width {msg.width}, height: {msg.height} \n")
        self.point_cloud_msg = msg

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
        
    def Start(self):
        while True:
            try:
                print(f"recieve {self.point_cloud_msg.header.frame_id} with number: {self.point_cloud_msg.width}")
            except:
                print("no point cloud data received yet.")
            
            try:
                print(f"recieve {self.height_map_msg_debug.frame_id} {self.height_map_data[0]}")
            except:
                print("no height map data received yet.")
            time.sleep(0.05)

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
