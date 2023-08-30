from template_matching import *
from extracting_color_spaces import *
from color_calibration import *
import cv2 as cv
from collections import defaultdict
import numpy as np
import datetime
import pickle



class ImageAdaptator:
    def __init__(self, index, dir_path, result_dir_path, main_path) -> None:
        '''
            Parameters:
                    inp  (ndarray): input image
                    ref  (ndarray): reference image
                    mask (ndarray): segmentation mask of reference image
                    index    (int): index of experiment to save data
                    dir_path (str): path to the directory to save results
        '''
        self.index = index
        self.dir_path = dir_path
        self.result_dir_path = result_dir_path
        self.main_path = main_path
        
    def DrawColorSpaceAndSave(self, colors, name):
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

    
    def GetRefColorSpace(self, images, masks) -> np.array:
        colors = []
        print(len(images))
        for i in range(len(images)):
            ref_image = images[i]
            ref_mask = masks[i][:, :, 0]
            colors_extractor = ColorSpacesExtractor(dir_path=self.dir_path)
            ref_colors = colors_extractor.extract_colors_from_image(ref_image, ref_mask, i + 1, ref_images= True)
            colors.append(ref_colors)
        colors = np.array(colors)
        ref_cs = np.nanmean(colors, axis=0)
        self.DrawColorSpaceAndSave(ref_cs)
        cs_dict = self.GetColorspaceDictionary(ref_cs)
        with open(f'{self.main_path}/colorspace.pkl', 'wb') as f:
            pickle.dump(cs_dict, f)
        return cs_dict
    
    def GetColorspaceDictionary(self, colors, ref_images = False):
        dict_colors = {}
        for i in range(len(colors)):
            color = colors[i][0][0]
            if not np.isnan(color):
                if ref_images:
                    dict_colors[i] = colors[i]
                else:
                    dict_colors[i-1] = colors[i]
        print(dict_colors)
        return dict_colors


    def CalibrateAlone(self, img, ref, mask, match_template=False) -> np.array:
        print('===============================================')
        input_image = img
        ref_image = ref
        ref_mask = mask[:,:,0]

        print(f'||INFO||: Unique minerals in mask {np.unique(ref_mask)}\n')

        if match_template:
            print(f'\n||INFO||: Template matching started {self.index}')
            
            template_matcher = TemplateMatcher(input_image, ref_image, ref_mask, self.index, self.dir_path)
            res_matcher = template_matcher.get_matched_templates()
            if (res_matcher == None):
                return
            inp_crops, ref_crops, mask_crops = res_matcher            
            print(f'||INFO||: Template matching finished {self.index}\n')
        else:
            inp_crops, ref_crops, mask_crops = [input_image], [ref_image],[ref_mask]       

        start_time_alone = datetime.datetime.now()

        print(f'\n||INFO||: Extracting colorspaces started {self.index}')
        colors_extractor = ColorSpacesExtractor(input_image, ref_image, 
                                                inp_crops, ref_crops, mask_crops, 
                                                self.index, self.dir_path)
        inp_colors, ref_colors = colors_extractor.extract_colors(True)
        print(f'||INFO||: Extracting colorspaces finished {self.index}\n')
        print(f'\n||INFO||: Color calibration started {self.index}')
        colors_calibrator = ColorCalibrator(input_image, 
                                            inp_colors, ref_colors, self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            ccm_shape='4x3',
                                            distance='de00',
                                            gamma = 2,
                                            linear = 'gamma')
        (result_image, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)

        print('||STATISTICS||: Working time alone: ', datetime.datetime.now() - start_time_alone)
        print(f'||INFO||: ColorCalibration finished {self.index}\n')
        print('===============================================')
        return ccm
    
    def GetColorSpace(self, img, mask, ref_dict) -> dict:
        print('===============================================')
        input_image = img
        mask = mask[:,:,0]

        print(f'||INFO||: Unique minerals in mask {np.unique(mask)}\n')     

        print(f'\n||INFO||: Extracting colorspaces started {self.index}')
        colors_extractor = ColorSpacesExtractor(dir_path=self.dir_path)
        inp_colors = colors_extractor.extract_colors_from_image(input_image, mask, self.index)
        print(f'||INFO||: Extracting colorspaces finished {self.index}\n')
        print(f'\n||INFO||: Color calibration started {self.index}')

        inp_dict = self.GetColorspaceDictionary(inp_colors)
        inp_colors_final = {}
        print(ref_dict)
        for i in inp_dict.keys():
            print(i, i in ref_dict.keys())
            if i in ref_dict.keys():
                inp_colors_final[i] = inp_dict[i]       
        return inp_colors_final
    
    def GetRefAndInputCS(self, ref_dict, inp_dicts_all):
        ref_colors = []
        inp_colors_final = []
        print(inp_dicts_all)
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
        self.DrawColorSpaceAndSave(inp_colors_final, "inp")
        self.DrawColorSpaceAndSave(ref_colors, "ref")
        ref_colors = np.squeeze(ref_colors, axis=1)
        inp_colors_final = np.squeeze(inp_colors_final, axis=1)
        return inp_colors_final, ref_colors

    def CalibrateWithoutRefImg(self, img, mask, ref_dict) -> np.array:
        print('===============================================')
        input_image = img
        mask = mask[:,:,0]

        print(f'||INFO||: Unique minerals in mask {np.unique(mask)}\n')     

        start_time_alone = datetime.datetime.now()

        print(f'\n||INFO||: Extracting colorspaces started {self.index}')
        colors_extractor = ColorSpacesExtractor(dir_path=self.dir_path)
        inp_colors = colors_extractor.extract_colors_from_image(input_image, mask, self.index)
        print(f'||INFO||: Extracting colorspaces finished {self.index}\n')
        print(f'\n||INFO||: Color calibration started {self.index}')

        inp_dict = self.GetColorspaceDictionary(inp_colors)
        ref_colors = []
        inp_colors_final = []
        print(ref_dict)
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
        (result_image, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)

        print('||STATISTICS||: Working time alone: ', datetime.datetime.now() - start_time_alone)
        print(f'||INFO||: ColorCalibration finished {self.index}\n')
        print('===============================================')
        return ccm
    
    def CalibrateWithColorSpaces(self, img, inp_cs, ref_cs) -> np.array:
        print('===============================================')
        input_image = img

        colors_calibrator = ColorCalibrator(input_image, 
                                            inp_cs, ref_cs, self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            ccm_shape='4x3',
                                            distance='de00',
                                            gamma = 2,
                                            linear = 'gamma')
        (result_image, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)

        print(f'||INFO||: ColorCalibration finished {self.index}\n')
        print('===============================================')
        return ccm

    def GetMaskPercentage(self, mask) -> int:
        h, w, _ = mask.shape
        not_zero = np.sum(mask[:,:,0] != 0)
        return not_zero / (h * w)

    def GetResultCCM(self, masks, ccms) -> np.array:
        percentages = []
        for mask in masks:
            percentages.append(self.GetMaskPercentage(mask))
        percentages = np.array(percentages)
        percentages = percentages / np.sum(percentages)
        return np.sum(ccms * percentages[:,None,None], axis=0)

    def InferImage(self, img, ccm):
        print('===============================================')
        print(f'\n||INFO||: Applying color correction matrix {self.index}')

        start_time_apply = datetime.datetime.now()

        colors_calibrator = ColorCalibrator(img, None, 
                                            None, self.index, 
                                            self.result_dir_path,
                                            self.main_path,
                                            only_infer=True)

        result_image = colors_calibrator.apply_calibrating_mask(ccm)
        
        print('||STATISTICS||: Working time apply: ', datetime.datetime.now() - start_time_apply)


        print(f'||INFO||: Applying CCM finished {self.index}\n')
        print('===============================================')
        return result_image
