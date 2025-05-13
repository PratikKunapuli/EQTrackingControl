import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from math import sin, cos
import numpy as np
import os

import sys

from ament_index_python.packages import get_package_share_directory

path = Path()

class MinimalPublisher(Node):

    def __init__(self, ref_pos, ref_quat, baseline_pos, baseline_quat, eq_pos, eq_quat, real_time_factor = 0.5, frame_rate = 20.0):
        super().__init__('visualizer')
        self.frame_rate = frame_rate

        # latching_qos = QoSProfile(depth=1,durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # self.reference_description_publisher_ = self.create_publisher(String, '/reference/robot_description', 10, qos_profile=latching_qos)
        # self.baseline_description_publisher_  = self.create_publisher(String, '/baseline/robot_description', 10, qos_profile=latching_qos)
        # self.symmetry_description_publisher_  = self.create_publisher(String, '/symmetry/robot_description', 10, qos_profile=latching_qos)
        # self.reference_description_publisher_ = self.create_publisher(String, '/reference/robot_description', 10)
        # self.baseline_description_publisher_  = self.create_publisher(String, '/baseline/robot_description', 10)
        # self.symmetry_description_publisher_  = self.create_publisher(String, '/symmetry /robot_description', 10)

        timer_period = 1.0/(frame_rate * real_time_factor)  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.time = 0
        self.index = 0

        self.ref_pos = ref_pos.astype(float) # (2000, 3)
        self.ref_quat = ref_quat.astype(float) # (2000, 4)
        self.baseline_pos = baseline_pos.astype(float) # (2000, 3)
        self.baseline_quat = baseline_quat.astype(float) # (2000, 4)
        self.eq_pos = eq_pos.astype(float) # (2000, 3)
        self.eq_quat = eq_quat.astype(float) # (2000, 4)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.path_publisher_ref = self.create_publisher(Path, '/path_ref', 10)
        self.path_publisher_eq = self.create_publisher(Path, '/path_eq', 10)
        self.path_publisher_baseline = self.create_publisher(Path, '/path_baseline', 10)
        self.path_ref = Path()
        self.path_eq = Path()
        self.path_baseline = Path()

        package_share_directory = get_package_share_directory('trajectory_visualization')


        print(package_share_directory)


        # msg = String()

        # with open(package_share_directory+'/astrobee_reference.urdf', 'r') as file:
        #     msg.data = file.read().replace('\n', '')
        # self.reference_description_publisher_.publish(msg)
        # with open(package_share_directory+'/astrobee_baseline.urdf', 'r') as file:
        #     msg.data = file.read().replace('\n', '')
        # self.baseline_description_publisher_.publish(msg)
        # with open(package_share_directory+'/astrobee_symmetry.urdf', 'r') as file:
        #     msg.data = file.read().replace('\n', '')
        # self.symmetry_description_publisher_.publish(msg)



    def timer_callback(self):

        now = self.get_clock().now().to_msg()

        self.get_logger().info('Time t=%f Index i=%d' % (self.time, self.index))
        

        # TODO: replace all these p's and q's with real data!
        t = self.time

        pose_ref = PoseStamped()
        pose_eq = PoseStamped()
        pose_baseline = PoseStamped()
        # publish all transforms
        self.send_pose('reference/body',self.ref_pos[self.index],self.ref_quat[self.index], now)
        self.send_pose('symmetry/body',self.eq_pos[self.index],self.eq_quat[self.index], now)
        self.send_pose('baseline/body',self.baseline_pos[self.index],self.baseline_quat[self.index], now)


        pose_ref.header.stamp = now
        pose_ref.header.frame_id = 'world'
        pose_ref.pose.position.x = self.ref_pos[self.index][0]
        pose_ref.pose.position.y = self.ref_pos[self.index][1]
        pose_ref.pose.position.z = self.ref_pos[self.index][2]
        pose_ref.pose.orientation.x = self.ref_quat[self.index][0]
        pose_ref.pose.orientation.y = self.ref_quat[self.index][1]
        pose_ref.pose.orientation.z = self.ref_quat[self.index][2]
        pose_ref.pose.orientation.w = self.ref_quat[self.index][3]

        pose_eq.header.stamp = now
        pose_eq.header.frame_id = 'world'
        pose_eq.pose.position.x = self.eq_pos[self.index][0]
        pose_eq.pose.position.y = self.eq_pos[self.index][1]
        pose_eq.pose.position.z = self.eq_pos[self.index][2]
        pose_eq.pose.orientation.x = self.eq_quat[self.index][0]
        pose_eq.pose.orientation.y = self.eq_quat[self.index][1]
        pose_eq.pose.orientation.z = self.eq_quat[self.index][2]
        pose_eq.pose.orientation.w = self.eq_quat[self.index][3]

        pose_baseline.header.stamp = now
        pose_baseline.header.frame_id = 'world'
        pose_baseline.pose.position.x = self.baseline_pos[self.index][0]
        pose_baseline.pose.position.y = self.baseline_pos[self.index][1]
        pose_baseline.pose.position.z = self.baseline_pos[self.index][2]
        pose_baseline.pose.orientation.x = self.baseline_quat[self.index][0]
        pose_baseline.pose.orientation.y = self.baseline_quat[self.index][1]
        pose_baseline.pose.orientation.z = self.baseline_quat[self.index][2]
        pose_baseline.pose.orientation.w = self.baseline_quat[self.index][3]

        self.path_ref.header.stamp = now
        self.path_ref.header.frame_id = 'world'
        self.path_ref.poses.append(pose_ref)

        self.path_eq.header.stamp = now
        self.path_eq.header.frame_id = 'world'
        self.path_eq.poses.append(pose_eq)

        self.path_baseline.header.stamp = now
        self.path_baseline.header.frame_id = 'world'
        self.path_baseline.poses.append(pose_baseline)


        if len(self.path_ref.poses) > 75:
            del self.path_ref.poses[0] # keep the path length fixed
        if len(self.path_eq.poses) > 75:
            del self.path_eq.poses[0] # keep the path length fixed
        if len(self.path_baseline.poses) > 75:
            del self.path_baseline.poses[0] # keep the path length fixed


        self.path_publisher_ref.publish(self.path_ref)
        self.path_publisher_eq.publish(self.path_eq)
        self.path_publisher_baseline.publish(self.path_baseline)

        self.time += 1.0 / self.frame_rate
        self.index += 1
        if self.index >= 2000:
            self.index = 0



    def send_pose(self, frame, p, q, now):

        t = TransformStamped()

        t.header.stamp = now
        t.header.frame_id = 'world'
        t.child_frame_id = frame

        t.transform.translation.x = p[0]
        t.transform.translation.y = p[1]
        t.transform.translation.z = p[2]

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # t.transform.rotation.x = q[1]
        # t.transform.rotation.y = q[2]
        # t.transform.rotation.z = q[3]
        # t.transform.rotation.w = q[0]
        
        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

def main():
    robot = sys.argv[2]
    print("visualizing the " + robot + "...")
    ref_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/ref_pos.npy").astype(np.float32)
    ref_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/ref_quat.npy").astype(np.float32)
    baseline_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/baseline_pos.npy").astype(np.float32)
    baseline_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/baseline_quat.npy").astype(np.float32)
    eq_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/equivariant_pos.npy").astype(np.float32)
    eq_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/equivariant_quat.npy").astype(np.float32)
    rclpy.init(args=None)

    print("Reference Start: ", ref_pos[0])
    print("EQ Start: ", eq_pos[0])
    print("Baseline Start: ", baseline_pos[0])
    input()


    minimal_publisher = MinimalPublisher(ref_pos, ref_quat, baseline_pos, baseline_quat, eq_pos, eq_quat)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()