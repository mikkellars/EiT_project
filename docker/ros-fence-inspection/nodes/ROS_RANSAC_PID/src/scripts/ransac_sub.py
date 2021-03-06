#!/usr/bin/env python3
import numpy as np

#ignore dumb errors
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn import linear_model
import rospy
import cv2 as cv
from sensor_msgs.msg import LaserScan
from dist_ransac.msg import Polar_dist
import time
import math
import time

MIN_RANGE = 0.4             #Meters
MAX_RANGE = 5.6             #Meters
MIN_INLIERS = 8             #Observations
RESIDUAL_THRESHOLD = 0.1    #Meters
MAX_FAILS = 1               #Nr of times RANSAC may fail before we give up
MAX_CLUSTER_DIST = 0.4      #Meters, distance between points in distinct clusters
CRITICAL_DIST = 0.6         #Meters, distance before prioritizing left lines

def nearest_point_on_line(line_start, line_end, point=np.array((0,0))):
    a_to_p = -line_start
    a_to_b = line_end - line_start

    a_to_b_magnitude = np.linalg.norm(a_to_b)

    if (a_to_b_magnitude == 0):
        return line_start

    a_to_b_unit = a_to_b/a_to_b_magnitude

    a_to_p_scaled = a_to_p * (1.0 / a_to_b_magnitude)

    #find how far along the line the point is
    t = np.dot(a_to_b_unit, a_to_p_scaled)
    if t < 0.0:
        t = 0
    elif t > 1.0:
        t = 1
            
    nearest = a_to_b * t + line_start

    return nearest


    #noise
def add_noise(points, n):
    return np.concatenate((points, (np.random.uniform(low=-msg.range_max, high=msg.range_max, size=(n,2)))))

#split the dataset into clusters naively
def naive_clustering(data, max_cluster_distance):
    clusters = []
    cluster_start = 0
    if data.shape[0] < 2:
        return data

    for i in range(1, data.shape[0]):
        if np.linalg.norm(data[i] - data[i-1]) > max_cluster_distance:
            #if the new cluster has only the point i, add that point as a singleton cluster
            if i == cluster_start + 1:
                clusters.append(np.expand_dims(data[i-1], axis=0))
            #otherwise, end the cluster normally
            else:
                clusters.append(data[cluster_start:i])
            cluster_start = i

        if i == data.shape[0] - 1:
            clusters.append(data[cluster_start:])
            break

    if clusters == []:
        clusters = np.array([data], dtype=object)
    else:
        clusters = np.array(clusters, dtype=object)

    return clusters

def naive_merge_cluster(clusters, max_cluster_distance):
    i = 0
    while (i < len(clusters)):
        j = i + 1
        while (j < len(clusters)):
            last = clusters[i][-1]
            first = clusters[j][0]
            if np.linalg.norm(first-last) < max_cluster_distance:
                try:
                    clusters[i] = np.concatenate((np.squeeze(clusters[i]), np.squeeze(clusters[j])), axis=0)
                    clusters = np.delete(clusters, j)
                    j -= 1
                except:
                    donothing = 0
                    #print("Strange concatenation error:")
                    #print("shapes:", np.squeeze(clusters[i].shape), np.squeeze(clusters[j].shape))
                    #print("points:", clusters[i], clusters[j])
            j+= 1
        i += 1
    return clusters

class RANSAC_subscriber():
    def __init__(self, simulate):
        self.simulate = simulate
        
        s_topic = "/laser/scan" 
        p_topic = "laser/dist_to_wall"
        if not self.simulate:
            s_topic = "/scan"

        self.subscription = rospy.Subscriber(s_topic, LaserScan, self.RANSAC, queue_size=1)
        print('starting RANSAC node')
        self.publisher = rospy.Publisher(p_topic, Polar_dist, queue_size=1)
      
        self.max_range = MAX_RANGE
        self.min_range = MIN_RANGE
        self.min_inliers = MIN_INLIERS
        self.residual_threshold = RESIDUAL_THRESHOLD
        self.max_fails = MAX_FAILS
        self.max_cluster_dist = MAX_CLUSTER_DIST
        self.image = np.array([0])
        self.drawScale = 125
        self.num = 0
        self.dists = []
        self.start_time = time.time()

       # with open('/assets/images/laser_scan/dist_err.npy', 'wb') as f:
       #    pass

    def RANSAC(self, msg):
        #start_time = time.time()
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_inc = msg.angle_increment
        ranges = np.array(msg.ranges)

        if len(ranges) == 0:
            raise(IOError, "NO POINTS FROM SCANNER")
        angle_arr = np.arange(angle_min, angle_max+(0.1*angle_inc), angle_inc)

        def y_dist(angle, dist):
            return np.sin(angle)*dist

        def x_dist(angle, dist):
            return np.cos(angle)*dist

        positions = np.array([np.array([x_dist(a, d), y_dist(a, d)]) for (a, d) in zip(angle_arr, ranges) if d < self.max_range and d > self.min_range])

        if len(positions) == 0:
            print("NO VALID POSITIONS")
            return

        self.image = np.zeros([np.int(np.ceil(self.drawScale*2*msg.range_max)),
                               np.int(np.ceil(self.drawScale*2*msg.range_max)), 3], dtype=np.uint8)
        self.draw_points(positions)

        #ctime = time.time()
        #print(positions.shape, " points")
        clusters = naive_clustering(positions, self.max_cluster_dist)
        #print(clusters.shape, "clusters")
        clusters = naive_merge_cluster(clusters, self.max_cluster_dist)
        #print(clusters.shape, "merged clusters")
        #print("clustering took", time.time()-ctime)

        rtime = time.time()
        # do a ransac
        fit_sets = []
        fit_models = []
        c = 0

        ri = 0
        while c < clusters.shape[0]:
            points = clusters[c]
            while np.array(points).shape[0] > self.min_inliers:
                fails = 0
                try:
                    ri += 1
                    rs = linear_model.RANSACRegressor(#stop_n_inliers=self.min_inliers,
                                                      residual_threshold=self.residual_threshold,
                                                      max_trials=10)
                    rs.fit(np.expand_dims(points[:, 0], axis=1), points[:, 1])
                    inlier_mask = rs.inlier_mask_
                    inlier_points = points[np.array(inlier_mask)]

                    #print(inlier_points.shape)
                    if  inlier_points.shape[0] < self.min_inliers:
                        fails += 1

                    inlier_splits = naive_clustering(inlier_points, self.max_cluster_dist)
                    #print("inlier sets of ", inlier_splits.shape)

                    for split in inlier_splits:
                        min_x = np.min(split[:,0], axis=0)
                        max_x = np.max(split[:,0], axis=0)
                        start = np.array([min_x, rs.predict([[min_x]])[0]])
                        end = np.array([max_x, rs.predict([[max_x]])[0]])
                        fit_sets.append(split)
                        fit_models.append(np.array([start, end]))

                    points = points[~np.array(inlier_mask)]

                    # #split the remaining points again
                    # new_clusters = np.array(naive_clustering(points, self.max_cluster_dist), dtype=np.object)
                    # if new_clusters.shape[0] > 1:
                    #     for nc in new_clusters:
                    #         if nc.shape[0] > self.min_inliers:
                    #             clusters = np.concatenate((clusters, []))
                    #             clusters[-1] = nc
                    #             break

                except:
                    fails += 1

                if fails >= self.max_fails:
                    break
            c += 1
        #print("RANSAC took", time.time() - rtime, "     and ran ", ri, "times")

        self.draw_lines(fit_models, fit_sets)

        min_dist = np.inf
        min_dist_point = np.array([0, 0])
        min_angle = np.inf

        # POINT ON LINE METHOD
        for model in fit_models:    
            #find nearest point on the line, relative to the robot
            point = nearest_point_on_line(model[0], model[1]) 

            #get the distance to the point
            dist = np.linalg.norm(point)

            if dist < min_dist:
                angle = math.atan2(point[1], point[0])
                #If  the wall is to the right or within critical distance
                if angle <= 0 or dist <= CRITICAL_DIST:
                    min_dist = dist
                    min_dist_point = point

        #if we dont find anything, dont publish
        if min_dist == np.inf:
            return

        min_angle = math.atan2(min_dist_point[1], min_dist_point[0])

        rmsg = Polar_dist()
        rmsg.dist = min_dist
        rmsg.angle = min_angle
        self.publisher.publish(rmsg)

        #time and dist logging
        time_diff = time.time()-self.start_time
        self.dists.append([time_diff, min_dist])

        cv.putText(self.image, f'Dist: {rmsg.dist:0.4f}, Angle: {np.rad2deg(rmsg.angle):0.4f}', (10,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))

        cx = np.int(np.round(self.image.shape[0]//2 + self.drawScale * min_dist_point[0]))
        cy = np.int(np.round(self.image.shape[1]//2 - self.drawScale * min_dist_point[1]))

        cv.circle(self.image, (cx,cy), 0, (255, 255, 255), thickness=3)

        saverate = 5
    #    if self.simulate:
    #        if (self.num % saverate == 0):
    #            cv.imwrite(f'/assets/images/laser_scan/scan_{self.num//10:03d}.jpg', self.image)
    #            print(f'Writing image: {self.num // saverate}')

    #    elif not self.simulate:
    #        cv.imwrite(f'/assets/images/laser_scan/scan_{self.num:03d}.jpg', self.image)
    #        print(f'Writing image: {self.num}')

        # if (not self.simulate and self.num % 1 == 0) or (self.simulate and self.num%500 == 0):
        #      with open('/assets/images/laser_scan/dist_err.npy', 'wb') as f:
        #         np.save(f, np.asarray(self.dists), allow_pickle=True)

        self.num += 1
        #print(f'Took { time.time() - start_time:0.3f} s')


    def draw_points(self, points):
        for point in points:
            try:
                cx = np.int(np.round(self.image.shape[0]/2 + self.drawScale * point[0]))
                cy = np.int(np.round(self.image.shape[1]/2 - self.drawScale * point[1]))
                #self.image[cx, cy] = (0, 0, 255)
                cv.circle(self.image, (cx, cy), 0, (0, 0, 255))
        #  x, -y
            except:
                print("Point draw err ", point)
        cv.arrowedLine(self.image, (self.image.shape[0]//2-2, self.image.shape[1]//2), (self.image.shape[0]//2+4, self.image.shape[1]//2), (0, 255, 0), 4)

    def draw_lines(self, lines, inliers):
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 255, 255)]
        ci = 0
        for line, points in zip(lines, inliers):
            color = colors[ci]
            ci += 1
            ci = ci % len(colors)
            #print(ci)
            for point in points:
                cx = np.int(np.round(self.image.shape[0]/2 + self.drawScale * point[0]))
                cy = np.int(np.round(self.image.shape[1]/2 - self.drawScale * point[1]))
                cv.circle(self.image, (cx, cy), 0, color)
            sx = np.int(np.round(self.image.shape[0]/2 + self.drawScale * line[0, 0]))
            sy = np.int(np.round(self.image.shape[0]/2 - self.drawScale * line[0, 1]))
            ex = np.int(np.round(self.image.shape[0]/2 + self.drawScale * line[1, 0]))
            ey = np.int(np.round(self.image.shape[0]/2 - self.drawScale * line[1, 1]))
            cv.line(self.image, (sx, sy), (ex, ey), color)

# def main(args=None):
#     RANSAC_node = RANSAC_subscriber()
#     rospy.spin()


# if __name__ == '__main__':
#     main()

