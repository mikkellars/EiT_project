#! /usr/bin/env python3

"""Script for spawning and evaluating ico learning in simulation
"""

import rospy
import random
import numpy as np

import roslaunch

from sim_spawn.spawn_frobit import SpawnFrobit
from ico.learn_and_follow import LearnFollow 

random.seed(0)

# Sqaure map spawn area
MAP_1 = np.array(([(9, -5), (8.5, -4.5)],[(9, -0.9),(8.5, -0.4)])) # [ [x-outer bound, x-inner bound], [y-outer bound, y-inner bound]] (9, -5), (8, -4)],[(9, -0.9),(8, 0.9)

def gen_spawn_pos(map_bound, num_of_pos):
    positions = list()
    pos_x = list()
    pos_y = list()
    cur_num = 0

    x_bound = map_bound[0]
    y_bound = map_bound[1]

    for i in range(num_of_pos):
        square_select = random.randint(0, 4)

        if square_select == 0:
            gen_x = random.uniform(x_bound[0][0], x_bound[1][0])
            gen_y = random.uniform(y_bound[0][0], y_bound[1][1])
            rand_ang = random.uniform(60, 120)
            # while(rand_ang > 60 and rand_ang < 120):
            #     rand_ang = random.uniform(0, 360)
        elif square_select == 1:
            gen_x = random.uniform(x_bound[0][0], x_bound[1][1])
            gen_y = random.uniform(y_bound[1][1], y_bound[0][1])
            rand_ang = random.uniform(-30, 30)
        elif square_select == 2:
            rand_ang = random.uniform(240, 300)
            gen_x = random.uniform(x_bound[1][1], x_bound[0][1])
            gen_y = random.uniform(y_bound[1][0], y_bound[0][1])
        elif square_select == 3:
            rand_ang = random.uniform(150, 210)
            gen_x = random.uniform(x_bound[1][0], x_bound[0][1])
            gen_y = random.uniform(y_bound[0][0], y_bound[1][0])

        
      #  positions.append((gen_x, gen_y, rand_ang))


    return positions

    

def main():
    # Launch file param
    # if not rospy.has_param('~simulate') or not rospy.has_param('~log'):
    #     ValueError('Need to set simulate and log param')

    # simulate = rospy.get_param('~simulate')
    # log = rospy.get_param('~log')

    # Generate poses for square map
    poses = gen_spawn_pos(MAP_1, 10)

    # Init for ico learning
    pub_name_mc = '/velocity_controller/cmd_vel'
    sub_name = 'laser/dist_to_wall' 

    target_dist = 1.0 # Meters
    learn_inteval = 0.2 # Meters: Accepted interval without learning

    rospy.init_node("ico_fence_follow_eval", anonymous=True)
    
    rospy.wait_for_service("/gazebo/spawn_urdf_model")

    spawn = SpawnFrobit()

    # Running ico learning on the different generate poses
    for x, y, angle in poses:
        spawn.move_frobit('frobit', x, y, angle=angle)
        
        # Running ico learning for one lap
        learn_follow = LearnFollow(sub_name, pub_name_mc, target_dist, learn_inteval, simulate=True, learn_type = 'one', log=False)
        while not spawn.completed_one_lap(time_before_check=5):
            pass

if __name__ == '__main__':
    main()
