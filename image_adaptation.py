from template_matching import *
from extracting_color_spaces import *
from color_calibration import *
from collections import defaultdict
from typing import Dict
import numpy as np
import datetime
import pickle



class ImageAdaptator:
    def __init__(self, index, dir_path, result_dir_path, main_path, 
                 logging = True, save_test_imgs = True) -> None:
        '''
        Parameters:
            inp  (ndarray): input image
            ref  (ndarray): reference image
            mask (ndarray): segmentation mask of reference image
            index    (int): index of experiment to save data
            dir_path (str): path to the directory to save results
            main_path(str): path to the root directory of current experiment
            logging (bool): flag, representing if logs are written in stdout  
            save_test_imgs: flag, representing if test images are saved
        '''
        self.index = index
        self.dir_path = dir_path
        self.result_dir_path = result_dir_path
        self.main_path = main_path
        self.logging = logging
        self.save_test_imgs = save_test_imgs
        
    def __drawColorSpaceAndSave(self, colors, name):
        colors_without_none = []
        for i in range(len(colors)):
            color = colors[i][0][0]
            if not np.isnan(color):
                colors_without_none += [colors[i]]
        real_size = len(colors_without_none)
        f = plt.subplots((math.ceil(np.sqrt(real_size))),
                         math.ceil(np.sqrt(real_size)), 
                         figsize=(15,15))[0]
        for i in range(real_size):
            image = np.zeros((300, 300, 3), np.uint8)
            image[:] = colors_without_none[i]
            f.axes[i].imshow(image)
        f.savefig(f'{self.dir_path}/{name}_color_space.png')

    
    def GetRefColorSpace(self, images, masks, name) -> Dict:
        '''
        Gets colorspace dict from given images with masks.

        Parameters:
            images (list of ndarrays): input images
            masks  (list of ndarrays): masks for input images
            name                (str): name of experiment, used to save colorspace image file
        '''
        colors = []
        for i in range(len(images)):
            ref_image = images[i]
            ref_mask = masks[i][:, :, 0]
            colors_extractor = ColorSpacesExtractor(dir_path=self.dir_path)
            ref_colors = colors_extractor.extract_colors_from_image(ref_image, ref_mask, i + 1, ref_images= True)
            colors.append(ref_colors)
        colors = np.array(colors)
        ref_cs = np.nanmean(colors, axis=0)
        if self.save_test_imgs: self.__drawColorSpaceAndSave(ref_cs, name)
        cs_dict = self.__getColorspaceDictionary(ref_cs)
        with open(f'{self.main_path}/colorspace.pkl', 'wb') as f:
            pickle.dump(cs_dict, f)
        return cs_dict
    
    def __getColorspaceDictionary(self, colors, ref_images = False) -> Dict:
        dict_colors = {}
        for i in range(len(colors)):
            color = colors[i][0][0]
            if not np.isnan(color):
                if ref_images:
                    dict_colors[i] = colors[i]
                else:
                    dict_colors[i-1] = colors[i]
        if self.logging: print("Retrieved colorspace = ", dict_colors)
        return dict_colors


    def CalibrateAlone(self, img, ref, mask, match_template=False) -> ndarray:
        '''
        Calibrate color correction model by given one input image, reference image and reference image mask.

        Parameters:
            img (ndarray): input image
            ref (ndarray): reference image
            mask(ndarray): mask for reference image
            match_template: flag, representing if input image is already matched with reference
        '''
        if self.logging: print('===============================================')
        input_image = img
        ref_image = ref
        ref_mask = mask[:,:,0]

        if self.logging: print(f'||INFO||: Unique minerals in mask {np.unique(ref_mask)}\n')

        if match_template:
            if self.logging: print(f'\n||INFO||: Template matching started {self.index}')
            
            template_matcher = TemplateMatcher(input_image, ref_image, ref_mask, self.index, self.dir_path)
            res_matcher = template_matcher.get_matched_templates()
            if (res_matcher == None):
                return np.zeros((4, 3))
            inp_crops, ref_crops, mask_crops = res_matcher            
            if self.logging: print(f'||INFO||: Template matching finished {self.index}\n')
        else:
            inp_crops, ref_crops, mask_crops = [input_image], [ref_image],[ref_mask]       

        start_time_alone = datetime.datetime.now()

        if self.logging: print(f'\n||INFO||: Extracting colorspaces started {self.index}')
        colors_extractor = ColorSpacesExtractor(input_image, ref_image, 
                                                inp_crops, ref_crops, mask_crops, 
                                                self.index, self.dir_path)
        inp_colors, ref_colors = colors_extractor.extract_colors(True)
        if self.logging: print(f'||INFO||: Extracting colorspaces finished {self.index}\n')
        if self.logging: print(f'\n||INFO||: Color calibration started {self.index}')
        colors_calibrator = ColorCalibrator(input_image, 
                                            inp_colors, ref_colors, self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            ccm_shape='4x3',
                                            distance='de00',
                                            gamma = 2,
                                            linear = 'gamma')
        (_, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)

        if self.logging: 
            print('||STATISTICS||: Working time alone: ', datetime.datetime.now() - start_time_alone)
            print(f'||INFO||: ColorCalibration finished {self.index}\n')
            print('===============================================')
        return ccm
    
    def GetColorSpace(self, img, mask, ref_dict = None) -> dict:
        '''
        Retrieves colorspace from input image by given mask.
        If ref_dict is specified returns only colors, which are in ref_dict.

        Parameters:
            img  (ndarray): input image
            mask (ndarray): mask for input image
            ref_dict(dict): flag, representing if input image is already matched with reference
        '''
        if self.logging: print('===============================================')
        input_image = img
        mask = mask[:,:,0]

        if self.logging: print(f'||INFO||: Unique minerals in mask {np.unique(mask)}\n')     

        if self.logging: print(f'\n||INFO||: Extracting colorspaces started {self.index}')
        colors_extractor = ColorSpacesExtractor(dir_path=self.dir_path)
        inp_colors = colors_extractor.extract_colors_from_image(input_image, mask, self.index)
        if self.logging: print(f'||INFO||: Extracting colorspaces finished {self.index}\n')
        if self.logging: print(f'\n||INFO||: Color calibration started {self.index}')

        inp_dict = self.__getColorspaceDictionary(inp_colors)
        inp_colors_final = {}
        if self.logging: print(ref_dict)
        for i in inp_dict.keys():
            if ref_dict is not None:
                print(i, i in ref_dict.keys())
                if i in ref_dict.keys():
                    inp_colors_final[i] = inp_dict[i]    
            else:   
                inp_colors_final[i] = inp_dict[i]    
        return inp_colors_final
    
    def GetRefAndInputCS(self, ref_dict, inp_dicts_all):
        '''
        Retrieves CS matrices from reference CS in dict and list of dicts with input CSs

        Parameters:
            ref_dict      (dict): reference colorspace in dict
            inp_dicts_all (dict): list of dicts with input colorspaces
        '''
        ref_colors = []
        inp_colors_final = []
        if self.logging: print("Input dicts = ", inp_dicts_all)
        by_color = defaultdict(list)
        for color_dict in inp_dicts_all:
            for index, color in color_dict.items():
                by_color[index].append(color)
        inp_dict = {index: sum(colors) / len(colors) for index, colors in by_color.items()}
        for i in inp_dict.keys():
            if i in ref_dict.keys():
                ref_colors += [ref_dict[i]]
                inp_colors_final += [inp_dict[i]]
        ref_colors = np.array(ref_colors)
        inp_colors_final = np.array(inp_colors_final)
        if self.save_test_imgs: 
            self.__drawColorSpaceAndSave(inp_colors_final, "inp")
            self.__drawColorSpaceAndSave(ref_colors, "ref")
        ref_colors = np.squeeze(ref_colors, axis=1)
        inp_colors_final = np.squeeze(inp_colors_final, axis=1)
        return inp_colors_final, ref_colors

    def CalibrateWithoutRefImg(self, img, mask, ref_dict) -> ndarray:
        '''
        Calibrate color correction model by given one input image, input image mask and reference colorspace in a dict.

        Parameters:
            img (ndarray): input image
            mask(ndarray): mask for input image
            ref_dict      (dict): reference colorspace in dict
        '''
        if self.logging: print('===============================================')
        input_image = img
        mask = mask[:,:,0]

        if self.logging: print(f'||INFO||: Unique minerals in mask {np.unique(mask)}\n')     

        start_time_alone = datetime.datetime.now()

        if self.logging: print(f'\n||INFO||: Extracting colorspaces started {self.index}')
        colors_extractor = ColorSpacesExtractor(dir_path=self.dir_path)
        inp_colors = colors_extractor.extract_colors_from_image(input_image, mask, self.index)
        if self.logging: print(f'||INFO||: Extracting colorspaces finished {self.index}\n')
        if self.logging: print(f'\n||INFO||: Color calibration started {self.index}')

        inp_dict = self.__getColorspaceDictionary(inp_colors)
        ref_colors = []
        inp_colors_final = []

        for i in inp_dict.keys():
            if i in ref_dict.keys():
                ref_colors += [ref_dict[i]]
                inp_colors_final += [inp_dict[i]]
        ref_colors = np.array(ref_colors)
        inp_colors_final = np.array(inp_colors_final)

        ref_colors = np.squeeze(ref_colors, axis=1)
        inp_colors_final = np.squeeze(inp_colors_final, axis=1)

        colors_calibrator = ColorCalibrator(input_image, 
                                            inp_colors_final, ref_colors, 
                                            self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            ccm_shape='4x3',
                                            distance='de00',
                                            gamma = 2,
                                            linear = 'gamma')
        (_, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)

        if self.logging: 
            print('||STATISTICS||: Working time alone: ', datetime.datetime.now() - start_time_alone)
            print(f'||INFO||: ColorCalibration finished {self.index}\n')
            print('===============================================')
        return ccm
    
    def CalibrateWithColorSpaces(self, img, inp_cs, ref_cs) -> ndarray:
        '''
        Calibrate color correction model by given one input image, input and reference colorspaces in matrices.

        Parameters:
            img    (ndarray): input image
            inp_cs (ndarray): input colorspace matrix
            ref_cs (ndarray): reference colorspace matrix
        '''
        if self.logging: print('===============================================')
        input_image = img

        colors_calibrator = ColorCalibrator(input_image, 
                                            inp_cs, ref_cs, self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            ccm_shape='4x3',
                                            distance='de00',
                                            gamma = 2,
                                            linear = 'gamma')
        (_, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)

        if self.logging: 
            print(f'||INFO||: ColorCalibration finished {self.index}\n')
            print('===============================================')
        return ccm

    def __getMaskPercentage(self, mask) -> int:
        h, w, _ = mask.shape
        not_zero = np.sum(mask[:,:,0] != 0)
        return not_zero / (h * w)

    def GetResultCCM(self, masks, ccms) -> ndarray:
        '''
        Returns weighted mean of list of CCMs depending on partial masks.
        Weigths are higher for CCMs with the more filled mask.

        Parameters:
            masks  (list of ndarray): list of partial masks
            ccms   (list of ndarray): list of CCMs
        '''
        percentages = []
        for mask in masks:
            percentages.append(self.__getMaskPercentage(mask))
        percentages = np.array(percentages)
        percentages = percentages / np.sum(percentages)
        return np.sum(ccms * percentages[:,None,None], axis=0)

    def InferImage(self, img, ccm):
        '''
        Applies given color correction matrix to the input image.

        Parameters:
            img    (ndarray): input image
            ccm    (ndarray): color correction matrix
        '''
        if self.logging: print('===============================================')
        if self.logging: print(f'\n||INFO||: Applying color correction matrix {self.index}')

        start_time_apply = datetime.datetime.now()

        colors_calibrator = ColorCalibrator(img, None, 
                                            None, self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            only_infer=True)

        result_image = colors_calibrator.apply_calibrating_mask(ccm)
        
        if self.logging: 
            print('||STATISTICS||: Working time apply: ', datetime.datetime.now() - start_time_apply)

            print(f'||INFO||: Applying CCM finished {self.index}\n')
            print('===============================================')
        return result_image
