# -*- coding: utf-8 -*-
"""
source: https://scikit-image.org/
source: http://www.ijicic.org/ijicic-140417.pdf

Created on Fri Sep  9 15:53:47 2022

@author: codyh
"""

from skimage import io
from skimage.feature import corner_harris, corner_peaks, corner_subpix
from skimage.transform import rescale
from skimage.filters.thresholding import try_all_threshold, threshold_li
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
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
        self._raw_img = None
    
    def set_img(self,img_path:str) -> list:
        self._raw_img = io.imread(img_path)
    
    def get_img(self) -> list:
        if self._raw_img.all() == None:
            raise ImageProcessorError('No image set!')
        return self._raw_img
    
    def get_img_size(self, img:list) -> tuple:
        height = np.size(img,0)
        width = np.size(img,1)
        return (width,height)
    
    def crop_img(self, pt1:tuple, pt2:tuple, img:list) -> list:
        #pt1 top left corner
        #pt2 bottom right corner
        
        x1,y1 = pt1
        x2,y2 = pt2
        
        return img[y1:y2+1,x1:x2+1]
    
    def constrain_img_size(self,img: list) -> list:
        width, height = self.get_img_size(img)

        if width > self.img_max_width:
            ratio = self.img_max_width / img.shape[1]
            return rescale(img, ratio)
        else:
            return img
    
    def detect_corners(self,constrain_img_size:bool = True, min_distance:int = 10,
    threshold_rel:float = 0.02, window_size:int = 5) -> list:
        raw_img = self._raw_img

        #convert to gray img and apply thresholding
        raw_img = rgb2gray(raw_img)
        thresh = threshold_li(raw_img)
        raw_img = raw_img > thresh
        
        if constrain_img_size:
            img = self.constrain_img_size(raw_img)
        else:
            img = raw_img
        
        coords = corner_peaks(corner_harris(img), min_distance=min_distance, threshold_rel=threshold_rel)
        corners = corner_subpix(img, coords, window_size=window_size)

        #sometimes corners returns NaN values in array, we must filter those.
        corners_nan = np.isnan(corners)
        corners_mask = []
        for value in corners_nan:
            corners_mask.append(value[0])
        return np.delete(corners,corners_mask,0)

    def get_center_point(self,corners: list) -> tuple:
        #center of mass for points in circles.
        #return corners.mean(axis=0)[0]
        num_points = len(corners)
        sum_x = 0.0
        sum_y = 0.0

        for point in corners:
            sum_x += point[1]
            sum_y += point[0]
        
        x = sum_x / num_points
        y = sum_y / num_points
        return x, y

    def get_points_distance_from_center(self,corners:list,center_pnt:tuple):
        points_with_dist = []
        center_x, center_y = center_pnt
        
        for corner in corners:
            dist = np.sqrt(pow((corner[1] - center_x),2) + pow((corner[0] - center_y),2))
            corner_point_with_dist = np.append(corner,dist)
            points_with_dist.append(corner_point_with_dist)
        
        points_with_dist = np.array(points_with_dist)
        #Sort points by distance in ascending order
        sorted_points = points_with_dist[points_with_dist[:,2].argsort()]
        return sorted_points

    def get_grouped_points(self,sorted_points:list, group_size:int):
        grouped_points = []
        
        groups = (len(sorted_points) // group_size)
        for i in range(groups):
            starting_group_index = i * group_size
            ending_group_index = starting_group_index + group_size
            grouped_points.append(sorted_points[starting_group_index:ending_group_index])
        
        return grouped_points
    
    def get_avg_deviations(self,grouped_points:list) -> list:
        deviations, radii = [], []

        #get average radii
        for circle in grouped_points:
            radii.append(circle.mean(axis=0)[2])

        for i,circle in enumerate(grouped_points):
            sum_deviations = 0.0
            for point in circle:
                sum_deviations += abs(point[2] - radii[i])
            avg_deviation = sum_deviations / len(circle)
            pct_deviation = avg_deviation / radii[i]
            deviations.append(pct_deviation)
        
        return deviations

    def draw_points(self, grouped_points:list, center_pnt:tuple,constrain_img_size:bool=True):
        center_x, center_y = center_pnt
        raw_img = self._raw_img
        dir_name = os.path.dirname(__file__)
        output_dir = '../../output/'
        pic_name = 'output.png'
        file_path = output_dir + pic_name
        output_path = os.path.join(dir_name,file_path)

        if constrain_img_size:
            img = self.constrain_img_size(raw_img)
        else:
            img = raw_img

        fig, ax = plt.subplots()
        width, height = self.get_img_size(img)

        ax.imshow(img,cmap=plt.cm.gray)
        colors = ['green', 'red', 'blue', 'purple', 'yellow', 'cyan', 'pink']
        ax.plot(center_x,center_y,'+r',markersize=10)
        for i,circles in enumerate(grouped_points):
            for points in circles:
                #ax.plot(points[1], points[0],'o', color=colors[i], markersize=8)
                ax.plot(points[1], points[0],'o', markersize=8)

        ax.axis((0, width, height, 0))
        #plt.show()
        fig.savefig(output_path)



if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    file_dir = '../../test_imgs/circle_mirror/'
    file_name = 'mirror17-cropped.jpg'
    file_path = file_dir + file_name
    constrain_img_size = True

    img_path = os.path.join(dir_name,file_path)

    img = io.imread(img_path)
    img = rgb2gray(img)
    thresh = threshold_li(img)
    img = img > thresh

    # fig, ax = try_all_threshold(img,figsize=(20, 12), verbose=False)
    # plt.show()

    imgproc = ImageProcessor()


    corners = imgproc.detect_corners(img,constrain_img_size=constrain_img_size)

    center_point = imgproc.get_center_point(corners)

    points_with_distance = imgproc.get_points_distance_from_center(corners,center_point)

    grouped_points = imgproc.get_grouped_points(points_with_distance,8)

    print(imgproc.get_avg_deviations(grouped_points))

    imgproc.draw_points(img, grouped_points, center_point,constrain_img_size=constrain_img_size)