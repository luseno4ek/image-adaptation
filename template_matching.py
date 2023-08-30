from distutils.log import error
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


class TemplateMatcher:
    def __init__(self, inp, ref, mask, index, dir_path) -> None:
        '''
            Parameters:
                    inp  (ndarray): input image
                    ref  (ndarray): reference image
                    mask (ndarray): segmentation mask of reference image
                    index    (int): index of experiment to save data
                    dir_path (str): path to the directory to save results
        '''
        self.index = index
        self.TEMPLATE_SIZE = 800
        self.templates_horizontally = mask.shape[1] // self.TEMPLATE_SIZE 
        self.templates_vertically = mask.shape[0] // self.TEMPLATE_SIZE 
        self.input_img_gray = cv.cvtColor(inp, cv.COLOR_BGR2GRAY)
        self.ref_img_gray = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
        self.MIN_MATCH_COUNT = 2
        self.input_img = inp
        self.ref_img = ref
        self.mask = mask
        self.dir_path = dir_path
        
    def check_coords(self, coord) -> bool:
        '''
        Check if given coordinates represent quadrilateral.

                Parameters:
                        coord (ndarray with shape (4,1,2)): coordinates of corners

                Returns:
                        true : coordinates represent quadrilateral
                        false: strange coordinates
        '''
        coord[coord < 0] = 0
        left_up = coord[0][0]
        left_down = coord[1][0]
        right_down = coord[2][0]
        right_up = coord[3][0]
        if ((left_up[0] > right_up[0]) or (left_up[0] > right_down[0]) 
            or (left_up[1] > left_down[1]) or (left_up[1] > right_down[1])):
            print("||INFO||: checking coords failed")
            return False
        else:
            if ((left_down[1] < right_up[1]) or (left_down[0] > right_down[0]) 
                or (left_down[0] > right_up[0])):
                print("||INFO||: checking coords failed")
                return False
            else:
                if (right_down[1] < right_up[1]):
                    print("||INFO||: checking coords failed")
                    return False
                else:
                    return True

    def get_matched_coordinates(self, temp_img, map_img):
        """
        Gets template and map image and returns matched coordinates in map image

            Parameters:
                    temp_img (ndarray): image to be used as template
                    map_img  (ndarray): image to be searched in

            Returns:
                    (dst, M): tuple with transformed coordinates and homography matrix
        """
        # initiate SIFT detector
        sift = cv.SIFT_create(  
                                nfeatures = 0, 
                                nOctaveLayers = 3,
                                contrastThreshold = 0.09,
                                edgeThreshold = 5,
                                sigma = 3
                                )

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(temp_img, None)
        kp2, des2 = sift.detectAndCompute(map_img, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        print(f"||INFO||: extracted {len(des1)} points from temp and {len(des2)} point from ref")

        # find matches by knn which calculates point distance in 128 dim
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        print(f"||INFO||: matched points count = {len(good)}")


        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # find homography
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 10, confidence=0.7)
            print(f"||INFO||: homography matrix = {M}")
            matchesMask = mask.ravel().tolist()

            h, w = temp_img.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                            [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)  # matched coordinates

            map_img = cv.polylines(
                map_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

        else:
            matchesMask = None
            return None

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matchesMask,  # draw only inliers
                        flags=2)

        # draw template and map image, matches, and keypoints
        img3 = cv.drawMatches(temp_img, kp1, map_img, kp2,
                            good, None, **draw_params)

        # result image path
        plt.imsave(f'{self.dir_path}/img{self.index}_template_matching_result.png', img3)

        return dst, M

    def match_templates(self) -> list:
        '''
        Implements SIFT template matching ref image with input.

                Returns:
                        good_templates (list): list of tuples 
                        (
                        first_template_index (int),
                        second_template_index (int),
                        transformed_coords (ndarray with shape (4, 1, 2)),
                        homography_matrix (ndarray with shape (3, 3))
                        ) 
        '''
        try:
            print("||INFO||: >>>>>>>>>>>>> matching full images <<<<<<<<<<<<<<<<<")
            eq_ref = cv.equalizeHist(self.ref_img_gray)
            eq_input = cv.equalizeHist(self.input_img_gray)
            coord_full, M_full = self.get_matched_coordinates(eq_ref, eq_input)
            print("||INFO||: received coordinates and matrix for full image")
            if (self.check_coords(coord_full)):
                print('|SUCCESS| --Template Matching--:  SIFT matching succed with full images')
                return [(-1, -1, coord_full, M_full)]
            else:
                print('!WARNING! --Template Matching--:  SIFT matching failed with full images')
                raise error

        except:
            print("||INFO||: >>>>>>>>>>>>> matching parts of images <<<<<<<<<<<<<<<<<")
            good_templates = []


            for i in range(self.templates_horizontally):
                for j in range(self.templates_vertically):
                    temp_img_gray = self.ref_img_gray[i*self.TEMPLATE_SIZE: (i+1)*self.TEMPLATE_SIZE, 
                                                    j*self.TEMPLATE_SIZE : (j+1)*self.TEMPLATE_SIZE]

                    temp_img_eq = cv.equalizeHist(temp_img_gray)

                    map_img_eq = cv.equalizeHist(self.input_img_gray)

                    try:
                        coord, M = self.get_matched_coordinates(temp_img_eq, map_img_eq)
                    except:
                        print('!WARNING! --Template Matching--:  SIFT matching failed in position i = {i}, j = {j}'.format(i=i,j=j))
                    else:
                        if (self.check_coords(coord)):
                            good_templates.append((i, j, coord, M))
                            print('|SUCCESS| --Template Matching--:  SIFT matching succed in position i = {i}, j = {j}'.format(i=i,j=j))
                        else:
                            print('!WARNING! --Template Matching--:  SIFT matching failed in position i = {i}, j = {j}'.format(i=i,j=j))
            return good_templates
    
    def create_matched_templates(self, good_templates) -> tuple:
        '''
        Applies homography transformation and gets matched templates.

                Parameters:
                    good_templates (list): result of match_templates method

                Returns:
                        templates (tuple): tuple of lists 
                        (
                        inp_crops (list of images),
                        ref_crops (list of images),
                        mask_crops (list of images)
                        ) 
        '''
        mask_crops = []
        ref_crops_rgb = []
        input_crops_rgb = []            

        for i in range(len(good_templates)):
            curr_data = good_templates[i]
            curr_i = curr_data[0]
            curr_j = curr_data[1]
            coord = curr_data[2]
            M = curr_data[3]

            if (curr_i == -1):
                mask_template = self.mask
                rgb_template = self.ref_img
            else:
                mask_template = self.mask[curr_i*self.TEMPLATE_SIZE : (curr_i+1)*self.TEMPLATE_SIZE, 
                                        curr_j*self.TEMPLATE_SIZE : (curr_j+1)*self.TEMPLATE_SIZE]
                rgb_template = self.ref_img[curr_i*self.TEMPLATE_SIZE : (curr_i+1)*self.TEMPLATE_SIZE, 
                                            curr_j*self.TEMPLATE_SIZE : (curr_j+1)*self.TEMPLATE_SIZE]
            MASK_UNIQUE = np.unique(mask_template)

            transformed_im = cv.warpPerspective(rgb_template, M, 
                                                dsize=(self.input_img.shape[1], self.input_img.shape[0]))[:,:,0:3]
            

            transformed_mask = cv.warpPerspective(mask_template, M, 
                                                  dsize=(self.input_img.shape[1], self.input_img.shape[0]))[:,:,0:3]
            MASK_UNIQUE = np.unique(transformed_mask)

            down_y = np.array([coord[1][0][1], coord[2][0][1]])
            up_y = np.array([coord[0][0][1], coord[3][0][1]])

            left_x = np.array([coord[0][0][0], coord[1][0][0]])
            right_x = np.array([coord[2][0][0], coord[3][0][0]])

            a = round(left_x.max())
            b = round(up_y.max())
            c = round(right_x.min())
            d = round(down_y.min())

            cropped_transformed_template = transformed_im[b:d,a:c]
            cropped_transformed_mask = transformed_mask[b:d,a:c][:,:,0]
            cropped_input_image = self.input_img[b:d,a:c]

            mask_crops.append(cropped_transformed_mask)
            ref_crops_rgb.append(cropped_transformed_template)
            input_crops_rgb.append(cropped_input_image)

        if (len(good_templates) > 0):
            n = len(good_templates)
            print(f'|SUCCESS| --Template Matching--: Matching is finished. Found {n} matching templates')
            f = plt.subplots(int(math.ceil(np.sqrt(n))),int(math.ceil(np.sqrt(n))), figsize=(15,15))[0]
            for i in range(n):
                f.axes[i].imshow(ref_crops_rgb[i])
            f.savefig(f'{self.dir_path}/img{self.index}_ref_image_crops.png')
            for i in range(n):
                f.axes[i].imshow(input_crops_rgb[i])
            f.savefig(f'{self.dir_path}/img{self.index}_inp_image_crops.png')
            return (input_crops_rgb, ref_crops_rgb, mask_crops)
        else:
            print('!!!ERROR!!! --Template Matching--:   Matching failed')
            return None

    def get_matched_templates(self):
        '''
        Returnes list of matched templates. Summarize all class work. 
        Saves plt graphics with matched templates.

                Returns:
                        templates (tuple): tuple of lists 
                        (
                        inp_crops (list of images),
                        ref_crops (list of images),
                        mask_crops (list of images)
                        ) 
        '''
        return self.create_matched_templates(self.match_templates())
