#! /usr/bin/env python3

"""Script for spawning the frobit model.
Inspiration from:
- https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/spawn_object.py 
- https://answers.ros.org/question/42849/how-to-launch-a-launch-file-from-python-code/
"""

import rospy
import roslib

import os
import subprocess
import math
from gazebo_msgs.srv import *
from gazebo_msgs.msg import ModelState 
from geometry_msgs.msg import Pose, Point
from scipy.spatial.transform import Rotation

class SpawnFrobit:
    def __init__(self, x:float = 0.0, y:float = 0.0, z:float = 0.0, via_launch_file:bool = True):
        self.x = x
        self.y = y
        self.z = z

        if not via_launch_file:
            self.pkg_location = self.__get_location()
            self.xml_string = self.__get_xml_cmd()
            self.srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        else:
            self.p = None

      #  self.delete_model('frobit')

        self.last_check_time = rospy.get_time()

    def __get_location(self):
        """Gets location of frobit packages created

        Returns:
            [type]: location of the packages
        """
        try:
            location = roslib.packages.get_pkg_dir('frobit_description') + '/urdf/frobit.xarco'
        except:
            ValueError('File not found: frobit_description/urdf/frobit.xarco')

        return location

    def __get_xml_cmd(self):
        """Get xml string for robot_description

        Returns:
            [type]: xml string
        """
        p = os.popen("rosrun xacro xacro " + self.pkg_location)
        xml_string = p.read()
        p.close()

        return xml_string
       
    def delete_model(self, name:str):
        """Deletes an existing model in the enviroment

        Args:
            name (str): name of the model
        """
        srv_delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
		# req = DeleteModelRequest()
		# req.model_name = name
        try:
            res = srv_delete_model(name)
        except e:
            print("Model %s does not exist in gazebo.", name)

    def spawn_model(self, name:str, x:float, y:float, z:float=0.0, angle:float=0.0):
        """Spawning the frobit at the coordinates

        Args:
            x (float): x-coordinate of robot spawn
            y (float): y-coordinate of robot spawn
            z (float): z-coordinate of robot spawn
        """
        # Updates position
        self.x = x
        self.y = y
        self.z = z

        # spawn new model
        req = SpawnModelRequest()
        req.model_name = name # model name from command line input
        req.model_xml = self.xml_string

        object_pose = Pose()
        object_pose.position.x = x
        object_pose.position.y = y
        object_pose.position.z = z
        rot = Rotation.from_euler('xyz', [0,0,angle], degrees=True)
        object_pose.orientation.x = rot.as_quat()[0]
        object_pose.orientation.y = rot.as_quat()[1]
        object_pose.orientation.z = rot.as_quat()[2]
        object_pose.orientation.w = rot.as_quat()[3]
        req.initial_pose = object_pose
        
        res = self.srv_spawn_model(req)
	
		# evaluate response
        if res.success == True:
            rospy.loginfo(res.status_message + " " + name)
        else:
            print("Error: model %s not spawn. error message = "% name + res.status_message)

    def spawn_via_launch(self, name:str, x:float, y:float, z:float=0.0, angle:float=0.0):
        """
        Does work as well from service/topic callbacks using launch files
        """    
        if self.p != None:
            self.p.kill()

        package = 'frobit_description'
        launch_file = 'spawn.launch'

        command = f"roslaunch {package} {launch_file} vel_cont_name:=test"#{0} {1}".format(package, launch_file)

        self.p = subprocess.Popen(command, shell=True)

        state = self.p.poll()

        if state is None:
            print("process is running fine")
        elif state < 0:
            print("Process terminated with error")
        elif state > 0:
            print("Process terminated without error")


    def move_frobit(self, name:str, x:float, y:float, z:float=0.0, angle:float=0.0):
        self.x = x
        self.y = y
        self.z = z

        state_msg = ModelState()
        state_msg.model_name = name
        state_msg.pose.position.x = self.x
        state_msg.pose.position.y = self.y
        state_msg.pose.position.z = self.z

        rot = Rotation.from_euler('xyz', [0,0,angle], degrees=True)
        state_msg.pose.orientation.x = rot.as_quat()[0]
        state_msg.pose.orientation.y = rot.as_quat()[1]
        state_msg.pose.orientation.z = rot.as_quat()[2]
        state_msg.pose.orientation.w = rot.as_quat()[3]

        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )

    def completed_one_lap(self, radius:float = 1.5, time_before_check:float = 5) -> bool:
        """Function to detect if the robot has completed one lap
        Uses a circle on the spawn point and if the robot is inside then a lap is completed.

        Args:
            radius (float, optional): Radius of the circle. Defaults to 0.5.

        Returns:
            complete_lap (bool): whether the robot complete one lap or not
        """
        complete_lap = False

        self.model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)
        self.object_coordinates = self.model_coordinates("frobit","")
        # Current model center
        x = self.object_coordinates.pose.position.x
        y = self.object_coordinates.pose.position.y
        
        # center of circle
        a = self.x
        b = self.y 
       
        if (x-a)**2 + (y-b)**2 <= (radius)**2:
            if rospy.get_time() - self.last_check_time > time_before_check:
                complete_lap = True
            self.last_check_time = rospy.get_time()

        return complete_lap
            