#! /usr/bin/env python3

"""
Live plotting from file.
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


if __name__ == "__main__":

    # Obstacle collision plot
    obs_col_fig = plt.figure()
    obs_col_fig.suptitle('Obstacle collisions detected')
    obs_col_fig.tight_layout()
    obs_col_ax1 = obs_col_fig.add_subplot(2,1,1)
    obs_col_ax2 = obs_col_fig.add_subplot(2,1,2)

    # Left right plot
    lr_fig = plt.figure()
    lr_fig.suptitle('Obstacle avoidance')
    lr_fig.tight_layout()
    lr_ax1 = lr_fig.add_subplot(3,1,1)
    lr_ax2 = lr_fig.add_subplot(3,1,2)
    lr_ax3 = lr_fig.add_subplot(3,1,3)

    # Difference plot
    human_fig = plt.figure()
    human_fig.suptitle('Human following')
    human_fig.tight_layout()
    human_ax1 = human_fig.add_subplot(3,1,1)
    human_ax2 = human_fig.add_subplot(3,1,2)
    human_ax3 = human_fig.add_subplot(3,1,3)

    # Human collision plot
    human_col_fig = plt.figure()
    human_col_fig.suptitle('Human collisions detected')
    human_col_fig.tight_layout()
    human_col_ax1 = human_col_fig.add_subplot(2,1,1)
    human_col_ax2 = human_col_fig.add_subplot(2,1,2)

    def animate(i):
        # --------------
        # Obstacle collision plot
        # --------------

        graph_data = open('pi_control/logs/obs_col.txt', 'r').read()
        lines = graph_data.split('\n')

        xs = list()
        col_lefts, col_rights = list(), list()
        
        for j, l in enumerate(lines):
            # if j == 60: break

            if len(l) > 1:
                x, col_left, col_right = l.split(',')

                # X
                xs.append(float(x))

                # Collision left and right
                col_lefts.append(float(col_left))
                col_rights.append(float(col_right))

        obs_col_ax1.clear()
        obs_col_ax1.plot(xs, col_lefts)
        obs_col_ax1.set_title('Left')
        obs_col_ax1.set_xticks([])

        obs_col_ax2.clear()
        obs_col_ax2.plot(xs, col_rights)
        obs_col_ax2.set_title('Right')

        # --------------------
        # Left right ICOs plot
        # --------------------

        graph_data = open('pi_control/logs/left_right.txt', 'r').read()
        lines = graph_data.split('\n')

        xs = list()
        in_lefts, in_rights = list(), list()
        w_lefts, w_rights = list(), list()
        o_lefts, o_rights = list(), list()
        
        for j, l in enumerate(lines):
            # if j == 60: break

            if len(l) > 1:
                x, in_left, in_right, w_left, w_right, o_left, o_right = l.split(',')

                # X
                xs.append(float(x))

                # Left right ICOs
                in_lefts.append(float(in_left))
                in_rights.append(float(in_right))
                w_lefts.append(float(w_left))
                w_rights.append(float(w_right))
                o_lefts.append(float(o_left))
                o_rights.append(float(o_right))
        
        # Input
        lr_ax1.clear()
        lr_ax1.plot(xs, in_lefts, label='left')
        lr_ax1.plot(xs, in_rights, label='right')
        lr_ax1.set_title('Input')
        lr_ax1.set_xticks([])
        lr_ax1.legend()

        # Weight
        lr_ax2.clear()
        lr_ax2.plot(xs, w_lefts, label='left')
        lr_ax2.plot(xs, w_rights, label='right')
        lr_ax2.set_title('Weight')
        lr_ax2.set_xticks([])
        lr_ax2.legend()

        # Output
        lr_ax3.clear()
        lr_ax3.plot(xs, o_lefts, label='left')
        lr_ax3.plot(xs, o_rights, label='right')
        lr_ax3.set_title('Output')
        lr_ax3.legend()

        # --------
        # Human ICOs
        # --------

        graph_data = open('pi_control/logs/human.txt', 'r').read()
        lines = graph_data.split('\n')

        xs = list()
        in_lefts, in_rights = list(), list()
        w_lefts, w_rights = list(), list()
        o_lefts, o_rights = list(), list()
        
        for j, l in enumerate(lines):
            # if j == 60: break

            if len(l) > 1:
                x, in_left, in_right, w_left, w_right, o_left, o_right = l.split(',')

                # X
                xs.append(float(x))

                # Left right ICOs
                in_lefts.append(float(in_left))
                in_rights.append(float(in_right))
                w_lefts.append(float(w_left))
                w_rights.append(float(w_right))
                o_lefts.append(float(o_left))
                o_rights.append(float(o_right))
        
        # Input
        human_ax1.clear()
        human_ax1.plot(xs, in_lefts, label='left')
        human_ax1.plot(xs, in_rights, label='right')
        human_ax1.set_title('Input')
        human_ax1.set_xticks([])
        human_ax1.legend()

        # Weight
        human_ax2.clear()
        human_ax2.plot(xs, w_lefts, label='left')
        human_ax2.plot(xs, w_rights, label='right')
        human_ax2.set_title('Weight')
        human_ax2.set_xticks([])
        human_ax2.legend()

        # Output
        human_ax3.clear()
        human_ax3.plot(xs, o_lefts, label='left')
        human_ax3.plot(xs, o_rights, label='right')
        human_ax3.set_title('Output')
        human_ax3.legend()

        # --------------
        # Human collision plot
        # --------------

        graph_data = open('pi_control/logs/human_col.txt', 'r').read()
        lines = graph_data.split('\n')

        xs = list()
        col_lefts, col_rights = list(), list()
        
        for j, l in enumerate(lines):
            # if j == 60: break

            if len(l) > 1:
                x, col_left, col_right = l.split(',')

                # X
                xs.append(float(x))

                # Collision left and right
                col_lefts.append(float(col_left))
                col_rights.append(float(col_right))

        human_col_ax1.clear()
        human_col_ax1.plot(xs, col_lefts)
        human_col_ax1.set_title('Left')
        human_col_ax1.set_xticks([])

        human_col_ax2.clear()
        human_col_ax2.plot(xs, col_rights)
        human_col_ax2.set_title('Right')

    ani = animation.FuncAnimation(obs_col_fig, animate, interval=1000)
    ani = animation.FuncAnimation(lr_fig, animate, interval=1000)
    ani = animation.FuncAnimation(human_fig, animate, interval=1000)
    ani = animation.FuncAnimation(human_col_fig, animate, interval=1000)
    
    plt.show()
