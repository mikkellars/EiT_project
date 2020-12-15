import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64

class DifferentialKinematics():
	def __init__(self, wheel_distance):
		self.wheel_dist = wheel_distance
		self.wheel_dist_half = self.wheel_dist / 2.0
                self.vel_lin = 0.0 
		self.vel_ang = 0.0 
		self.vel_left = 0.0 
		self.vel_right = 0.0 

	def forward (self, vel_left, vel_right):
		self.vel_lin = (vel_right + vel_left)/2.0 # [m/s]
		self.vel_ang = (vel_right - vel_left)/self.wheel_dist # [rad/s]
		return (self.vel_lin, self.vel_ang)

	def inverse (self, vel_lin, vel_ang):
		self.vel_left  = vel_lin - self.wheel_dist_half*vel_ang # [m/s]
		self.vel_right = vel_lin + self.wheel_dist_half*vel_ang # [m/s]
		return (self.vel_left, self.vel_right) 

class FrobitKinematics():
    def __init__(self):
        rospy.init_node('kinematics_node', anonymous=True)

        self.subscriber = rospy.Subscriber("/frobit/twist", TwistStamped, self.twistCallback)
        self.pub_left = rospy.Publisher("/frobit/left_setpoint", Float64, queue_size = 1)
        self.pub_right = rospy.Publisher("/frobit/right_setpoint", Float64, queue_size = 1)

        self.model = DifferentialKinematics(39.0)

    def twistCallback(self,msg):
        left_msg = Float64()
        right_msg = Float64()
        
        vel_left, vel_right = self.model.inverse(msg.twist.linear.x, msg.twist.angular.z)

        left_msg.data = vel_left
        right_msg.data = vel_right

        self.pub_left.publish(left_msg)
        self.pub_right.publish(right_msg)

    def spin(self):
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            print("All is well")
            r.sleep()

