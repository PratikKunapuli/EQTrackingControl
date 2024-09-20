import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from math import sin, cos
import numpy as np

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = cos(ai)
    si = sin(ai)
    cj = cos(aj)
    sj = sin(aj)
    ck = cos(ak)
    sk = sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


class MinimalPublisher(Node):

    def __init__(self, real_time_factor = 1.0, frame_rate = 20.0):
        super().__init__('visualizer')
        self.frame_rate = frame_rate
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 1.0/(frame_rate * real_time_factor)  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.time = 0

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

    def timer_callback(self):

        self.get_logger().info('t=%f' % self.time)
        self.time += 1.0 / self.frame_rate

        # TODO: replace all these p's and q's with real data!
        t = self.time

        p_ref = np.array([ cos(.1*t), sin(.3*t), .1 * sin(t) ])
        q_ref = quaternion_from_euler(t*.1, t*.2, t*.3)

        p_sym = p_ref + np.array([.1 * sin(t), .05 * cos(t), 0])
        q_sym = quaternion_from_euler((t+ 3*sin(t))*.1, (t + 3*sin(t))*.2, (t + 3*sin(t))*.3)

        p_base = p_ref + np.array([.1,.2, .3])
        q_base = q_sym

        # publish all transforms
        self.send_pose('reference/body',p_ref,q_ref)
        self.send_pose('symmetry/body',p_sym,q_sym)
        self.send_pose('baseline/body',p_base,q_base)

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

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()