import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from math import sin, cos
import numpy as np
import os

import sys

from ament_index_python.packages import get_package_share_directory

class MinimalPublisher(Node):

    def __init__(self, ref_pos, ref_quat, baseline_pos, baseline_quat, eq_pos, eq_quat, real_time_factor = 1.0, frame_rate = 20.0):
        super().__init__('visualizer')
        self.frame_rate = frame_rate

        # latching_qos = QoSProfile(depth=1,durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        # self.reference_description_publisher_ = self.create_publisher(String, '/reference/robot_description', 10, qos_profile=latching_qos)
        # self.baseline_description_publisher_  = self.create_publisher(String, '/baseline/robot_description', 10, qos_profile=latching_qos)
        # self.symmetry_description_publisher_  = self.create_publisher(String, '/symmetry/robot_description', 10, qos_profile=latching_qos)
        # self.reference_description_publisher_ = self.create_publisher(String, '/reference/robot_description', 10)
        # self.baseline_description_publisher_  = self.create_publisher(String, '/baseline/robot_description', 10)
        # self.symmetry_description_publisher_  = self.create_publisher(String, '/symmetry/robot_description', 10)

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

        self.get_logger().info('Time t=%f' % self.time)
        self.time += 1.0 / self.frame_rate
        self.index += 1
        if self.index >= 2000:
            self.index = 0

        # TODO: replace all these p's and q's with real data!
        t = self.time


        # publish all transforms
        self.send_pose('reference/body',self.ref_pos[self.index],self.ref_quat[self.index])
        self.send_pose('symmetry/body',self.eq_pos[self.index],self.eq_quat[self.index])
        self.send_pose('baseline/body',self.baseline_pos[self.index],self.baseline_quat[self.index])



    def send_pose(self, frame, p, q):

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
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
    ref_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/ref_pos.npy")
    ref_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/ref_rotm.npy")
    baseline_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/baseline_pos.npy")
    baseline_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/baseline_rotm.npy")
    eq_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/equivariant_pos.npy")
    eq_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/equivariant_rotm.npy")
    rclpy.init(args=None)

    minimal_publisher = MinimalPublisher(ref_pos, ref_quat, baseline_pos, baseline_quat, eq_pos, eq_quat)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()