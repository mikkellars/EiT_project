"""
Corner detection for fence inspection
"""


import os
import sys
sys.path.append(os.getcwd())

import time
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Corner detection for fence inspection')
    parser.add_argument('--data_dir', type=str, default='vision/data/fence_data/test_set/labels', help='path to data directory')
    args = parser.parse_args()
    return args


def main(args):
    
    files = [f'{args.data_dir}/{f}' for f in os.listdir(args.data_dir)]
    for f in files:
        original = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        height, width = original.shape
        img = original.copy()

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel)
        img = cv2.erode(img, kernel)

        # img = skeletonize(img)

        # tmp = np.zeros((height, width, 3), dtype=original.dtype)
        tmp = cv2.bitwise_not(img)
        cnts, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 5000]
        cv2.drawContours(tmp, cnts, -1, (255, 255, 255), 3)

        plt.imshow(tmp, cmap='gray'), plt.axis('off'), plt.show()


def skeletonize(im:np.array)->np.array:
    """Returns a skeletonized verson of the im.
    Reference: http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    Args:
        im (np.array): Input image.

    Returns:
        np.array: Skeletonized image.
    """
    im = im.copy()
    skel = im.copy()
    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.morphologyEx(im, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(im, temp)
        skel = cv2.bitwise_or(skel, temp)
        im[:, :] = eroded[:, :]
        if cv2.countNonZero(im) == 0: break
    return skel


class HoughBundler:
    """Clusterize and merget each cluster of cv2.HoughLinesP() output.
    Example of use:
        a = HoughBundler()
        foo = a.process_lines(houghP_lines, binary_image)
    """

    def get_orientation(self, line:np.array)->float:
        """Get orientation of line using its length.

        Args:
            line (np.array): Line.

        Returns:
            float: orientation of line in degrees.
        """
        orientation = math.atan2(abs((line[0]-line[2])), abs((line[1]-line[3])))
        return math.degrees(orientation)

    def distance_point_line(self, point:np.array, line:np.array)->float:
        """Get distance between point and line

        Args:
            point (np.array): Point.
            line (np.array): Line.

        Returns:
            float: Distance.
        """
        # px, py = point
        # x1, y1, x2, y2 = line

        # def line_magnitude(x1, y1, x2, y2):
        #     return math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1), 2))

        # line_mag = line_magnitude(x1, y1, x2, y2)
        # if line_mag < 1e-8:
        #     return 9999

        # u = (((px-x1)*(x2-x1)) + ((py-y1)*(y2-y1)))
        # u = u / (line_mag * line_mag)

        # if u < 1e-5 or u > 1:
        #     ix = line_magnitude(px, py, x1, y1)
        #     iy = line_magnitude(px, py, x2, y2)

        #     if ix > iy:
        #         distance_point_line = iy
        #     else:
        #         distance_point_line = ix
        # else:
        #     ix = x1 + u * (x2 - x1)
        #     iy = y1 + u * (y2 - y1)
        #     distance_point_line = line_magnitude(px, py, ix, iy)

        # return distance_point_line
        px, py = point
        x1, y1, x2, y2 = line
        x_diff = x2 - x1
        y_diff = y2 - y1
        num = abs(y_diff * px - x_diff * py + x2 * y1 - y2 * x1)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den

    def get_distance(self, a:np.array, b:np.array)->float:
        """Get all possible distances between each dot of two lines and second
        line and return the shortest.

        Args:
            a (np.array): Line a.
            b (np.array): Line b.

        Returns:
            float: Shortest distance between two lines.
        """
        dist1 = self.distance_point_line(a[:2], b)
        dist2 = self.distance_point_line(a[2:], b)
        dist3 = self.distance_point_line(b[:2], a)
        dist4 = self.distance_point_line(b[2:], a)
        return min(dist1, dist2, dist3, dist4)

    def checker(self, line_new:np.array, groups:list,
                min_dist_to_merge:int, min_angle_to_merge:int)->bool:
        """Check if line have enough distance and angle to be count as similar.

        Args:
            line_new (np.array): New line.
            groups (list): Groups of lines.
            min_dist_to_merge (int): Minimum distance to merge.
            min_angle_to_merge (int): Minimum angle to merge.

        Returns:
            bool: If true, a new line if found.
        """
        for group in groups:
            for line_old in group:
                if self.get_distance(line_old, line_new) < min_dist_to_merge:
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        return True

    def merge_lines_pipeline(self, lines:np.array)->list:
        """Clusterize lines into groups.

        Args:
            lines (np.array): Lines.

        Returns:
            list: list of lines grouped.
        """
        groups = list()
        min_dist_to_merge = 5
        min_angle_to_merge = 2
        groups.append([lines[0]])
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_dist_to_merge, min_angle_to_merge):
                groups.append([line_new])
        return groups

    def merge_lines_segments(self, lines)->list:
        """Sort lines cluster and return first and last coordinates.

        Args:
            lines ([type]): Lines.

        Returns:
            list: first and last point in sorted group.
        """
        orientation = self.get_orientation(lines[0])

        if len(lines) == 1:
            return [lines[0][:2], lines[0][2:]]

        points = list()
        for line in lines:
            points.append(line[2:])
            points.append(line[2:])
        
        if 45 < orientation < 135:
            points = sorted(points, key=lambda point: point[1])
        else:
            points = sorted(points, key=lambda point: point[0])

        return [points[0], points[-1]]

    def process_lines(self, lines:np.array, img:np.array)->list:
        """Process lines.

        Args:
            lines (np.array): cv2.HoughLinesP() output.
            img (np.array): binary image

        Returns:
            list: merged lines.
        """
        lines_x, lines_y = list(), list()
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            if 45 < orientation < 135:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = list()

        for i in [lines_x, lines_y]:
                if len(i) > 0:
                    groups = self.merge_lines_pipeline(i)
                    merged_lines = list()
                    for group in groups:
                        merged_lines.append(self.merge_lines_segments(group))

                    merged_lines_all.extend(merged_lines)

        return merged_lines_all


if __name__ == '__main__':
    print(__doc__)
    args = parse_arguments()
    start_time = time.time()
    main(args)
    end_time = time.time() - start_time
    print(f'Done! It took {end_time//60:.0f}m {end_time%60:.0f}s')
