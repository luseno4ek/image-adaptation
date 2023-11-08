from numpy import ndarray
from typing import Tuple
from ColorCalibrationModel import color
from ColorCalibrationModel import api
from ColorCalibrationModel import colorspace
from matplotlib import pyplot as plt

class ColorCalibrator:
    def __init__(self, inp, inp_colorspace, ref_colorspace, index, dir_path, main_path,
                ccm_shape='4x3',
                distance='de00',
                gamma = 2,
                linear = 'gamma', 
                only_infer = False,
                logging = True) -> None:
        '''
        Parameters:
            inp  (ndarray): input image
            inp_colorspace (ndarray): color space of input image
            ref_colorspace (ndarray): color space of reference image
            index    (int): index of experiment to save data
            dir_path (str): path to the directory to save results
            main_path(str): path to the root directory of current experiment
            logging (bool): flag, representing if logs are written in stdout
        '''
        self.index = index
        self.dir_path = dir_path
        self.main_path = main_path
        self.input_image = inp
        self.logging = logging
        if not only_infer:
            self.inp_cs = inp_colorspace
            self.ref_cs = color.Color(ref_colorspace / 255, colorspace.sRGB)  
            self.ccm_shape = ccm_shape
            self.distance = distance 
            self.linear = linear
            self.gamma = gamma
            self.deg = 2
            self.saturated_threshold = (0, 1)
            self.weights_list = None
            self.weights_coeff = 0
            self.initial_method = 'white_balance' 
            self.xtol = 1e-4
            self.ftol = 1e-4
            if self.index == 1:
                self.write_params()

    def write_params(self):
        with open(f'{self.main_path}\\CCM_params.txt', 'w') as f:
            f.write(f'{self.ccm_shape=}'.split('=')[0] + ' = ' + f'{self.ccm_shape}\n')
            f.write(f'{self.distance=}'.split('=')[0] + ' = ' + f'{self.distance}\n')
            f.write(f'{self.linear=}'.split('=')[0] + ' = ' + f'{self.linear}\n')
            f.write(f'{self.gamma=}'.split('=')[0] + ' = ' + f'{self.gamma}\n')
    
    def write_ccm(self, ccm):
        with open(f'{self.main_path}\\CCM_params.txt', 'a') as f:
            f.write(f'{self.index}.' + f'{ccm=}'.split('=')[0] + ' = ' + f'{ccm}\n')
    
    def get_calibrated_image(self, return_ccm = False) -> Tuple[ndarray, ndarray]:
        '''
        Calibrates input image color space.

        Returns:
            input_image_updated (ndarray)
        '''
        if self.logging: print("inp_cs = ", self.inp_cs, " ref_cs = ", self.ref_cs.colors)
        ccm = api.color_calibration(self.inp_cs / 255, self.ref_cs, 
            colorspace = colorspace.sRGB,
            ccm_shape = self.ccm_shape, 
            distance = self.distance, 
            linear = self.linear, gamma = self.gamma, deg = self.deg, 
            saturated_threshold = self.saturated_threshold, 
            weights_list = self.weights_list, weights_coeff = self.weights_coeff, 
            initial_method = self.initial_method, 
            xtol = self.xtol, ftol = self.ftol
            )

        self.write_ccm(ccm.ccm)

        new_input_image = ccm.infer_image(self.input_image)

        plt.imsave(f'{self.dir_path}/{self.index}.jpg', new_input_image)

        if return_ccm:
            return (new_input_image, ccm.ccm)

        return new_input_image

    def apply_calibrating_mask(self, user_ccm) -> ndarray:
        '''
        Calibrates input image color space using user CCM matrix.
    
        Returns:
            input_image_updated (ndarray)
        '''
        ccm = api.color_calibration(None, None, only_infer=True)

        ccm.ccm = user_ccm

        new_input_image = ccm.infer_image(self.input_image)

        plt.imsave(f'{self.dir_path}/{self.index}.jpg', new_input_image)

        return new_input_image