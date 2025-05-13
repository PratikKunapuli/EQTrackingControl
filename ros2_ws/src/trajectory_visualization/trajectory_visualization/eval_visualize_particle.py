import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseStamped, TransformStamped
from geometry_msgs.msg import PointStamped, Point32
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from math import sin, cos
import numpy as np
import os

import sys

from ament_index_python.packages import get_package_share_directory

path = Path()

class MinimalPublisher(Node):

    def __init__(self, ref_pos, baseline_pos, eq_pos, real_time_factor = 0.5, frame_rate = 20.0):
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

        self.ref_pos = ref_pos.astype(float) # (2000, 50, 3)
        self.baseline_pos = baseline_pos.astype(float) # (2000, 50, 3)
        self.eq_pos = eq_pos.astype(float) # (2000, 50, 3)

        # Initialize the transform broadcaster
        # self.tf_broadcaster = TransformBroadcaster(self)
        self.eq_pointcloud_publisher = self.create_publisher(PointCloud, '/eq_point_cloud', 10)
        self.baseline_pointcloud_publisher = self.create_publisher(PointCloud, '/baseline_point_cloud', 10)
        self.ref_pointcloud_publisher = self.create_publisher(PointCloud, '/ref_point_pos', 10)

        self.num_particles = 20 # self.eq_pos.shape[1]

        self.eq_point_cloud = PointCloud()
        self.baseline_point_cloud = PointCloud()
        self.ref_point_cloud = PointCloud()
        self.path_publisher_ref = self.create_publisher(Path, '/path_ref', 10)
        self.path_publishers_eq = [self.create_publisher(Path, '/path_eq' + str(i), 10) for i in range(self.num_particles)]
        self.path_publishers_baseline = [self.create_publisher(Path, '/path_baseline' + str(i), 10) for i in range(self.num_particles)]
        self.path_ref = Path() 
        self.path_eqs = [Path() for i in range(self.num_particles)]
        self.path_baselines = [Path() for i in range(self.num_particles)]

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
        #self.send_pose(self.ref_pos[:, self.index, :])

        self.eq_point_cloud.header.stamp = self.get_clock().now().to_msg()
        self.eq_point_cloud.header.frame_id = 'world'
        self.eq_point_cloud.points = [] 
        #print(self.eq_pos.shape)
        for i in range(self.num_particles):
            p = Point32()
            p.x = self.eq_pos[self.index, i, 0]
            p.y = self.eq_pos[self.index, i, 1]
            p.z = self.eq_pos[self.index, i, 2]
            
            self.eq_point_cloud.points.append(p)

        # if len(self.eq_point_cloud.points) > self.num_particles * 50:
        #     del self.eq_point_cloud.points[0:50]

        #del self.eq_point_cloud.points[0:len(self.eq_point_cloud.points)]

        self.baseline_point_cloud.header.stamp = self.get_clock().now().to_msg()
        self.baseline_point_cloud.header.frame_id = 'world'
        self.baseline_point_cloud.points = []
        for i in range(self.num_particles):
            p = Point32()
            p.x = self.baseline_pos[self.index, i, 0]
            p.y = self.baseline_pos[self.index, i, 1]
            p.z = self.baseline_pos[self.index, i, 2]
            
            self.baseline_point_cloud.points.append(p)

        # if len(self.baseline_point_cloud.points) > self.num_particles * 100:
        #     del self.baseline_point_cloud.points[0:50]

        self.ref_point_cloud.header.stamp = self.get_clock().now().to_msg()
        self.ref_point_cloud.header.frame_id = 'world'
        self.eq_point_cloud.header.stamp = self.get_clock().now().to_msg()
        self.eq_point_cloud.header.frame_id = 'world'

        ref_point = Point32()
        ref_point.x = self.ref_pos[self.index, 0, 0]
        ref_point.y = self.ref_pos[self.index, 0, 1]
        ref_point.z = self.ref_pos[self.index, 0, 2]
        self.ref_point_cloud.points = [ref_point] 

        self.eq_pointcloud_publisher.publish(self.eq_point_cloud)
        self.baseline_pointcloud_publisher.publish(self.baseline_point_cloud)
        self.ref_pointcloud_publisher.publish(self.ref_point_cloud)


        pose_ref = PoseStamped()

        pose_ref.header.stamp = self.get_clock().now().to_msg()
        pose_ref.header.frame_id = 'world'
        pose_ref.pose.position.x =  self.ref_pos[self.index, 0, 0]
        pose_ref.pose.position.y =  self.ref_pos[self.index, 0, 1]
        pose_ref.pose.position.z =  self.ref_pos[self.index, 0, 2]
        pose_ref.pose.orientation.x = 0.0
        pose_ref.pose.orientation.y = 0.0
        pose_ref.pose.orientation.z = 0.0
        pose_ref.pose.orientation.w = 1.0
        self.path_ref.header.stamp = self.get_clock().now().to_msg()
        self.path_ref.header.frame_id = 'world'
        self.path_ref.poses.append(pose_ref)

        if len(self.path_ref.poses) > 75:
            del self.path_ref.poses[0] # keep the path length fixed to 20

        self.path_publisher_ref.publish(self.path_ref)


        for i in range(self.num_particles):

            pose_eqs = [PoseStamped() for i in range(self.num_particles)]
            pose_baselines = [PoseStamped() for i in range(self.num_particles)]

            pose_eqs[i].header.stamp = self.get_clock().now().to_msg()
            pose_eqs[i].header.frame_id = 'world'
            pose_eqs[i].pose.position.x = self.eq_pos[self.index,i,0]
            pose_eqs[i].pose.position.y = self.eq_pos[self.index,i,1]
            pose_eqs[i].pose.position.z = self.eq_pos[self.index,i,2]
            pose_eqs[i].pose.orientation.x = 0.0
            pose_eqs[i].pose.orientation.y = 0.0
            pose_eqs[i].pose.orientation.z = 0.0
            pose_eqs[i].pose.orientation.w = 1.0

            pose_baselines[i].header.stamp = self.get_clock().now().to_msg()
            pose_baselines[i].header.frame_id = 'world'
            pose_baselines[i].pose.position.x = self.baseline_pos[self.index,i,0]
            pose_baselines[i].pose.position.y = self.baseline_pos[self.index,i,1]
            pose_baselines[i].pose.position.z = self.baseline_pos[self.index,i,2]
            pose_baselines[i].pose.orientation.x = 0.0
            pose_baselines[i].pose.orientation.y = 0.0
            pose_baselines[i].pose.orientation.z = 0.0
            pose_baselines[i].pose.orientation.w = 1.0

            self.path_eqs[i].header.stamp = self.get_clock().now().to_msg()
            self.path_eqs[i].header.frame_id = 'world'
            self.path_eqs[i].poses.append(pose_eqs[i])

            self.path_baselines[i].header.stamp = self.get_clock().now().to_msg()
            self.path_baselines[i].header.frame_id = 'world'
            self.path_baselines[i].poses.append(pose_baselines[i])

            if len(self.path_eqs[i].poses) > 75:
                del self.path_eqs[i].poses[0] # keep the path length fixed to 20
            if len(self.path_baselines[i].poses) > 75:
                del self.path_baselines[i].poses[0] # keep the path length fixed to 20


            self.path_publishers_eq[i].publish(self.path_eqs[i])
            self.path_publishers_baseline[i].publish(self.path_baselines[i])



    def send_pose(self, p):

        for i in range(len(p)):
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
    # ref_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/ref_pos.npy")
    # ref_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/ref_rotm.npy")
    # baseline_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/baseline_pos.npy")
    # baseline_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/baseline_rotm.npy")
    # eq_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/equivariant_pos.npy")
    # eq_quat = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/equivariant_rotm.npy")

    ref_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/anim_ref.npy")
    baseline_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/anim_baseline.npy")
    eq_pos = np.load("./src/trajectory_visualization/trajectory_visualization/data/" + robot + "/anim_eq.npy")

    rclpy.init(args=None)

    input()

    minimal_publisher = MinimalPublisher(ref_pos, baseline_pos, eq_pos,)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()