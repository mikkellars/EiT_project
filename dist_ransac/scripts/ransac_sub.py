#!/usr/bin/env python
import numpy as np
from sklearn import linear_model
import rospy
import cv2 as cv
from sensor_msgs.msg import LaserScan
from dist_ransac.msg import Polar_dist

class RANSAC_subscriber():
    def __init__(self):
        rospy.init_node("ransac_wall_dist_pub", anonymous=True)
        topic = "/laser/scan"
        self.subscription = rospy.Subscriber(topic, LaserScan, self.RANSAC)
        self.publisher = rospy.Publisher("laser/dist_to_wall", Polar_dist, queue_size=10)
        rate = rospy.Rate(10)  # or whatever
        self.image = np.array([0])
        self.drawScale = 25

    def RANSAC(self, msg):
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_inc = msg.angle_increment
        ranges = np.array(msg.ranges)
        #ranges = ranges[ranges > msg.range_min]
        #ranges = ranges[ranges < msg.range_max]
        if len(ranges) == 0:
            raise(IOError, "NO POINTS FROM SCANNER")
        angle_arr = np.arange(angle_min, angle_max+(0.1*angle_inc), angle_inc)

        def y_dist(angle, dist):
            return np.sin(angle)*dist

        def x_dist(angle, dist):
            return np.cos(angle)*dist

        positions = np.array([np.array([x_dist(a, d), y_dist(a, d)]) for (a, d) in zip(angle_arr, ranges)])
        positions = positions[np.isfinite(positions).any(axis=1)]
        if len(positions) == 0:
            raise(IOError, "NO IN-RANGE POINTS")

        #noise
        def add_noise(points, n):
            return np.concatenate((points, (np.random.uniform(low=-msg.range_max, high=msg.range_max, size=(n,2)))))
        positions = add_noise(positions, 25)

        self.image = np.zeros([np.int(np.ceil(self.drawScale*2*msg.range_max)),
                               np.int(np.ceil(self.drawScale*2*msg.range_max)), 3], dtype=np.uint8)
        self.draw_points(positions)

        # do a ransac
        fit_sets = []
        fit_models = []
        min_samples = max(positions.size/10, 20) #TUNE THIS
        min_inliers = 20
        while np.array(positions).shape[0] > min_samples:
            #try:
            rs = linear_model.RANSACRegressor(min_samples=min_samples)
            rs.fit(np.expand_dims(positions[:, 0], axis=1), positions[:, 1])
            inlier_mask = rs.inlier_mask_
            inlier_points = positions[np.array(inlier_mask)]
            if inlier_points.shape[0] < min_inliers:
                break
            min_x = np.min(inlier_points[:,0], axis=0)
            max_x = np.max(inlier_points[:,0], axis=0)
            start = np.array([min_x, rs.predict([[min_x]])[0]])
            end = np.array([max_x, rs.predict([[max_x]])[0]])
            fit_sets.append(inlier_points)
            fit_models.append(np.array([start, end]))
            positions = positions[~np.array(inlier_mask)]
            #except:
            #    break

        if (len(fit_models) == 0):
            raise(AssertionError, "NO LINES COULD BE FIT")
            return

        self.draw_lines(fit_models)

        def nearest_point_on_line(line_start, line_end):
            a_to_p = -line_start
            a_to_b = line_end - line_start
            sq_mag_a_to_b = a_to_b[0]**2 + a_to_b[1]**2
            if sq_mag_a_to_b == 0:
                print("LINE OF ZERO LENGTH")
                return np.array([0, 0])
            dot_product = a_to_p[0]*a_to_b[0] + a_to_p[1]*a_to_b[1]
            dist_a_to_c = dot_product / sq_mag_a_to_b
            c = np.array([start[0] + a_to_b[0]*dist_a_to_c, start[1] + a_to_b[1]*dist_a_to_c])
            return c

        min_dist = np.inf
        min_dist_point = np.array([0, 0])
        for sets in fit_models:
            point = nearest_point_on_line(sets[0], sets[1])
            boop = point[0]**2 + point[1]**2
            dist = np.sqrt(point[0]**2 + point[1]**2)
            if dist < min_dist:
                min_dist = dist
                min_dist_point = point

        def angle_to_point(point):
            if point[0] == 0:
                if point[1] > 0:
                    return np.pi/2
                else:
                    if point[1] < 0:
                        return -np.pi/2
                    else:
                        print("IN COLLISION")
                        return 0
            if point[0] < 0:
                a = np.pi + np.arctan(point[1] / point[0])
                if a > np.pi:
                    a = -2 * np.pi + a
                return a
            return np.arctan(point[1]/point[0])

        rmsg = Polar_dist()
        rmsg.dist = min_dist
        rmsg.angle = angle_to_point(min_dist_point)
        self.publisher.publish(rmsg)

        cv.imshow('image', self.image)
        cv.waitKey(1)

    def draw_points(self, points):
        for point in points:
            cx = np.int(np.round(self.image.shape[0]/2 + self.drawScale * point[0]))
            cy = np.int(np.round(self.image.shape[1]/2 - self.drawScale * point[1]))
            #self.image[cx, cy] = (0, 0, 255)
            cv.circle(self.image, (cx, cy), 0, (0, 0, 255))
        #  x, -y
        cv.arrowedLine(self.image, (self.image.shape[0]/2-2, self.image.shape[1]/2), (self.image.shape[0]/2+2, self.image.shape[1]/2), (0, 255, 0))

    def draw_lines(self, lines):
        for line in lines:
            sx = np.int(np.round(self.image.shape[0]/2 + self.drawScale * line[0, 0]))
            sy = np.int(np.round(self.image.shape[0]/2 - self.drawScale * line[0, 1]))
            ex = np.int(np.round(self.image.shape[0]/2 + self.drawScale * line[1, 0]))
            ey = np.int(np.round(self.image.shape[0]/2 - self.drawScale * line[1, 1]))
            cv.line(self.image, (sx, sy), (ex, ey), (255, 0, 0))

def main(args=None):
    RANSAC_node = RANSAC_subscriber()
    rospy.spin()


if __name__ == '__main__':
    main()