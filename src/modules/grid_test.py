"""
source: https://scikit-image.org/

"""

from skimage import io
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.transform import rescale
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import math
import numpy as np
import os

class ImageProcessorError(Exception):
    def __init__(self,*args):
        super().__init__(args)
        if args:
            self.msg = args[0]
        else:
            self.msg = None
    
    def __st__(self):
        if self.msg:
            return self.msg
        else:
            return 'ImageProcessorError'

class ImageProcessor():
    def __init__(self, img_max_width:int = 800):
        self.img_max_width = img_max_width
    
    def get_img_size(self, img:list) -> tuple:
        height = np.size(img,0)
        width = np.size(img,1)
        return (width,height)
    
    def scale_img(self, img:list) -> list:
        #get img ratio
        ratio = self.img_max_width / img.shape[1]

        return rescale(img, ratio)
    
    def crop_img(self, pt1:tuple, pt2:tuple, img:list) -> list:
        #pt1 top left corner
        #pt2 bottom right corner
        
        x1,y1 = pt1
        x2,y2 = pt2
        
        return img[y1:y2+1,x1:x2+1]
    
    def detect_corners(self, raw_img:list) -> list:
        width, height = self.get_img_size(raw_img)

        if width > self.img_max_width:
            img = self.scale_img(raw_img)
        else:
            img = raw_img
        
        coords = corner_peaks(corner_harris(img), min_distance=5, threshold_rel=0.02)
        corners = corner_subpix(img, coords, window_size=15)

        #sometimes corners returns NaN values in array, we must filter those.
        corners_nan = np.isnan(corners)
        corners_mask = []
        for value in corners_nan:
            corners_mask.append(value[0])
        return np.delete(corners,corners_mask,0)

    def find_center(self,corners:list) -> tuple:
        
        num_corners = len(corners)
        sum_x = 0.0
        sum_y = 0.0
        for points in corners:
            sum_x += points[1]
            sum_y += points[0]
        
        x = sum_x / num_corners
        y = sum_y / num_corners

        return x, y
    
    def get_points_dist_from_center(self,points:list,center_pnt:tuple) -> list:
        center_x, center_y = center_pnt
        points_with_dist = []

        for point in points:
            dist = math.sqrt(pow(abs(point[1] - center_x),2) + pow(abs(point[0] - center_y),2))
            point_dist = np.append(point,dist)
            points_with_dist.append(point_dist)
        
        points_with_dist = np.array(points_with_dist)
        return points_with_dist[points_with_dist[:, 2].argsort()]
    
    def group_points(self,points:list,inner_sqr_ptnum:int=28) -> list:
        grouped_points = []
        num_groups = 0
        num_points = len(points)
        group_size = []

        #groups and group sizes can be calculated if we know the number of points of the inner-most square
        #group size is inner_sqr_ptnum + 8 for each square after the inner-most one.
        while(num_points > 0):
            num_points -= inner_sqr_ptnum
            num_groups += 1
            group_size.append(inner_sqr_ptnum)
            inner_sqr_ptnum += 8
            #if points not divisible by inner_sqr_ptnum, catch-all in last group
            if num_points <= inner_sqr_ptnum:
                num_groups += 1
                group_size.append(num_points)
                break
        

        for i in range(num_groups):
            if i == 0:
                starting_group_index = 0
            else:
                starting_group_index = group_size[i - 1] + group_size[i - 2] if i > 1 else group_size[i - 1]
            end_group_index = starting_group_index + (group_size[i])
            grouped_points.append(points[starting_group_index:end_group_index])
        
        return grouped_points

        


    def draw_points(self, raw_img:list, grouped_points:list, center: tuple):

        width, height = self.get_img_size(raw_img)
        if width > self.img_max_width:
            img = self.scale_img(raw_img)
        else:
            img = raw_img

        fig, ax = plt.subplots()
        width, height = self.get_img_size(img)

        ax.imshow(img,cmap=plt.cm.gray)
        colors = ['green', 'red', 'blue']
        #ax.plot(grouped_points[:, 1], grouped_points[:, 0], 'o', markersize=8)
        for i, squares in enumerate(grouped_points):
            for points in squares:
                ax.plot(points[1], points[0],'o',color=colors[i], markersize=8)

        ax.plot(center[0], center[1], '+r', markersize=8)

        ax.axis((0, width, height, 0))
        plt.show()

if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    file_dir = '../../test_imgs/'
    file_name = 'grid_test.jpg'
    file_path = file_dir + file_name

    img_path = os.path.join(dir_name,file_path)

    img = io.imread(img_path)
    img = rgb2gray(img)

    imgproc = ImageProcessor()

    corners = imgproc.detect_corners(img)
    square_center = imgproc.find_center(corners)

    points_with_dist = imgproc.get_points_dist_from_center(corners,square_center)
    grouped_points = imgproc.group_points(points_with_dist)
    imgproc.draw_points(img,grouped_points,square_center)


