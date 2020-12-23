#! /usr/bin/env python3

"""Script for spawning the frobit model.
Inspiration from https://github.com/ipa320/srs_public/blob/master/srs_user_tests/ros/scripts/spawn_object.py 
"""

import rospy
import roslib

import os
import math
from gazebo_msgs.srv import *
from geometry_msgs.msg import Pose, Point
from scipy.spatial.transform import Rotation

class SpawnFrobit:
    def __init__(self, x:float = 0.0, y:float = 0.0, z:float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.pkg_location = self.__get_location()
        self.xml_string = self.__get_xml_cmd()
        self.srv_spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)

        self.delete_model('frobit')

    def __get_location(self):
        """Gets location of frobit packages created

        Returns:
            [type]: location of the packages
        """
        # location = None
        # try:
        location = roslib.packages.get_pkg_dir('frobit_description') + '/urdf/frobit.xarco'
        # except:
        #     ValueError('File not found: frobit_description/urdf/frobit.xarco')

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

    def spawn_model(self, name:str, x:float, y:float, z:float, angle:float):
        """Spawning the frobit at the coordinates

        Args:
            x (float): x-coordinate of robot spawn
            y (float): y-coordinate of robot spawn
            z (float): z-coordinate of robot spawn
        """
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