import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
from unitree_sdk2py.idl.unitree_arm.msg.dds_ import ArmString_
import math
from dataclasses import dataclass
import json
import mujoco
import mujoco.viewer as mjv
import numpy as np 
from pynput import keyboard
from pynput.keyboard import Key, KeyCode
import threading
import pickle
import inspect
from pathlib import Path 
class Custom:
    def __init__(self):
        current_path = Path(__file__).resolve().parent
        xml_path = current_path/"robots"/"unitree_go2"/"scene_unitree_arm.xml"
        self.traj_path = 'scalemmd_trajectory.pkl'
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.default_q = self.model.keyframe("home").qpos
        self.default_arm_joint_pos = self.model.keyframe("home").qpos[7:15]
        self.default_go2_joint_pos = self.model.keyframe("home").qpos[15:]
        self.current_q = self.default_q
        self.joint_mapping = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

    # Public methods
    def Init(self):
        self.sport_client = SportClient()  
        self.sport_client.SetTimeout(10.0)
        self.sport_client.Init()

        self.arm_subscriber = ChannelSubscriber("rt/arm_Feedback", ArmString_)
        self.arm_subscriber.Init(self.ArmMessageHandler, 10)

        self.arm_publisher = ChannelPublisher("rt/arm_Command", ArmString_)
        self.arm_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        self.low_state = None 

        self.highstate_subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        self.highstate_subscriber.Init(self.HighStateHandler, 10)
        self.high_state = None 

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

        self.pos_world = np.zeros(3) 
        self.quat_world = np.zeros(4)

        self.vel_scale_x = 0.3
        self.vel_scale_y = 0.3
        self.vel_scale_rot = 1.0
        self.tracking_vel = 0.3

        self.key_combo = {KeyCode.from_char('a'), # left
                          KeyCode.from_char('w'), # forward 
                          KeyCode.from_char('s'), # backward
                          KeyCode.from_char('d'), # right
                          KeyCode.from_char('q'), # rot left 
                          KeyCode.from_char('e'), # rot right
                          KeyCode.from_char('p'),} # stop

        self.currently_pressed = set()
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.stop = False

        with open(self.traj_path, 'rb') as file:
            self.traj_points = pickle.load(file)
        self.target_point_idx = 0

    def on_press(self, key):
        self.currently_pressed.add(key)
        if KeyCode.from_char('a') in self.currently_pressed:
            self.vy = self.vel_scale_y
        if KeyCode.from_char('w') in self.currently_pressed:
            self.vx = self.vel_scale_x
        if KeyCode.from_char('s') in self.currently_pressed:
            self.vx = -self.vel_scale_x
        if KeyCode.from_char('d') in self.currently_pressed:
            self.vy = -self.vel_scale_y
        if KeyCode.from_char('q') in self.currently_pressed:
            self.vyaw = self.vel_scale_rot
        if KeyCode.from_char('e') in self.currently_pressed:
            self.vyaw = -self.vel_scale_rot
        if KeyCode.from_char('p') in self.currently_pressed:
            self.stop = True

    def on_release(self, key):
        if key == KeyCode.from_char('w') or key ==  KeyCode.from_char('s'):
            self.vx = 0 
        if key == KeyCode.from_char('a') or key ==  KeyCode.from_char('d'):
            self.vy = 0 
        if key == KeyCode.from_char('q') or key ==  KeyCode.from_char('e'):
            self.vyaw = 0 
        self.currently_pressed.discard(key)

        if key == KeyCode.from_char('p'):
            return False

    def Start(self):
        self.sport_client.StandUp()
        input("Press Enter to unlock joint...")
        self.sport_client.BalanceStand()
        self.reset_arm()
        stop_event = threading.Event()
        wait_time = 0
        with mjv.launch_passive(self.model, self.data) as viewer:
            viewer_thread = threading.Thread(
                target=self.sync_viewer, args=(stop_event, viewer), daemon=True
                )
            viewer_thread.start()
            input("press any key to start control")
            while True:
                current_points =  self.traj_points[self.target_point_idx]
                next_points = self.traj_points[self.target_point_idx + 1]
                dx = next_points[0] - current_points[0]
                dy = next_points[1] - current_points[1]
                yaw_tar = np.arctan2(dy, dx)
                self.MoveToPos(current_points[0], current_points[1], yaw_tar)

                # move to next target util robot reach the current one 
                pos_error = np.linalg.norm(current_points[:2] - self.high_state.position[:2]) 

                if pos_error <= 0.15 or wait_time>=1.0:
                    self.target_point_idx += 1
                    wait_time = 0
                wait_time += 0.1
                time.sleep(0.1)
                
                if self.stop or self.target_point_idx==self.traj_points.shape[0]-2:
                    self.sport_client.StandDown()
                    time.sleep(2)
                    self.turn_off_arm()
                    self.sport_client.Damp()
                    self.listener.join()
                    break
    def Test(self):
        current_points =  self.traj_points[self.target_point_idx]
        next_points = self.traj_points[self.target_point_idx + 1]
        dx = next_points[0] - current_points[0]
        dy = next_points[1] - current_points[1]
        yaw_tar = np.arctan2(dy, dx)
        self.MoveToPos(current_points[0], current_points[1], yaw_tar)

    def MoveToPos(self, x_tar, y_tar, yaw_tar):
        diff_x = x_tar - self.high_state.position[0]
        diff_y = y_tar - self.high_state.position[1]
        
        # vel_norm = np.linalg.norm(np.array([diff_x, diff_y]))
        # vx = diff_x / vel_norm * self.vel_scale_x
        # vy = diff_y / vel_norm * self.vel_scale_y
        
        vx, vy = self.high_state.velocity[0], self.high_state.velocity[1]

        cmd_x = np.clip(1.5 * diff_x - 0.1 * vx, -self.vel_scale_x, self.vel_scale_x)
        cmd_y = np.clip(1.5 * diff_y - 0.1 * vy, -self.vel_scale_y, self.vel_scale_y)

        vyaw = np.clip(yaw_tar - self.high_state.imu_state.rpy[2], -self.vel_scale_rot, self.vel_scale_rot)
        self.sport_client.Move(cmd_x, cmd_y, vyaw)

    def sync_viewer(self, stop_event, viewer):
        # Initialize way points 
        cnt = 0
        self.traj_points[:, 2] = 0.3
        viewer.user_scn.ngeom = 0
        npoints = self.traj_points.shape[0]
        for i in range(npoints * 2 + 1):
            color = np.array([1.0 * i / (npoints - 1), 1.0, 0.0, 1.0])
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[cnt],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size = np.zeros(3),
                rgba=color,
                pos=[0,0,0],
                mat=np.eye(3).flatten(),
            )
            cnt += 1
        viewer.user_scn.ngeom = cnt
        for i in range(self.traj_points.shape[0]):
                # draw spheres
                mujoco.mjv_initGeom(viewer.user_scn.geoms[i],
                                    type = mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size = [0.02, 0.02, 0.02],
                                    pos = self.traj_points[i],
                                    mat = np.eye(3).flatten(),
                                    rgba = np.array([1.0 * i / (npoints - 1), 1.0, 0.0, 1.0]))
        
        for i in range(self.traj_points.shape[0]-1):
                # draw lines
                mujoco.mjv_connector(viewer.user_scn.geoms[self.traj_points.shape[0]+i],
                                    mujoco.mjtGeom.mjGEOM_LINE,
                                    0.1,
                                    self.traj_points[i], 
                                    self.traj_points[i+1])
        viewer.sync()


        while not stop_event.is_set() and viewer.is_running():
            self.data.qpos[:3] = self.high_state.position
            self.data.qpos[3:7] = self.high_state.imu_state.quaternion
            # self.data.qpos[7:15] = self
            for i in range(12):
                self.data.qpos[15+i] = self.low_state.motor_state[self.joint_mapping[i]].q
            
            # draw current target
            mujoco.mjv_initGeom(viewer.user_scn.geoms[npoints * 2],
                                type = mujoco.mjtGeom.mjGEOM_SPHERE,
                                size = [0.04, 0.04, 0.04],
                                pos = self.traj_points[self.target_point_idx],
                                mat = np.eye(3).flatten(),
                                rgba = np.array([1.0, 0.0, 0.0, 1.0]))
            
            mujoco.mj_forward(self.model, self.data)
            viewer.sync()
            time.sleep(0.05)
    
    def send_arm_command(self):
        return 

    def reset_arm(self):
        data = '{"seq":4,"address":1,"funcode":2,"data":{"mode":1,"angle0":0,"angle1":-60,"angle2":60,"angle3":0,"angle4":30,"angle5":0,"angle6":0}}'
        arm_cmd = ArmString_(data_=data)
        self.arm_publisher.Write(arm_cmd)

    def turn_off_arm(self):
        data = '{"seq":4,"address":1,"funcode":5,"data":{"mode":0}}'
        arm_cmd = ArmString_(data_=data)
        self.arm_publisher.Write(arm_cmd)
        
    def ArmMessageHandler(self, msg: ArmString_):
        # print(msg)
        obj = json.loads(msg.data_)
        data = obj.get("data","")
        # print("data is :", data)
        # print("msg is: ", msg)
    
    def HighStateHandler(self, msg: SportModeState_):
        self.high_state = msg

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg

if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()
