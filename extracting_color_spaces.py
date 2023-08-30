import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.cluster import KMeans
from scipy import stats


class ColorSpacesExtractor:
    def __init__(self, inp = None, ref = None, inp_crops = None, ref_crops = None, 
                 mask_crops = None, index = None, dir_path = None) -> None:
        '''
            Parameters:
                    inp  (ndarray): input image
                    ref  (ndarray): reference image
                    inp_crops (ndarray of images): crops of input image
                    ref_crops (ndarray of images): crops of reference image
                    mask_crops(ndarray of images): crops of segmentation mask of reference image
                    index    (int): index of experiment to save data
                    dir_path (str): path to the directory to save results
        '''
        self.index = index
        self.dir_path = dir_path
        self.input_image = inp
        self.ref_image = ref
        self.inp_crops = inp_crops
        self.ref_crops = ref_crops
        self.mask_crops = mask_crops
        self.unique_minerals = np.arange(0)
        self.unique_minerals_count = 0

    def get_unique_minerals_count(self):
        '''
        Counts unique minerals in masks templates.
        Fills self.unique_minerals and self.unique_minerals_count.
        '''
        for mask_crop in self.mask_crops:
            MASK_UNIQUE = np.unique(mask_crop)
            self.unique_minerals = np.unique(np.concatenate((np.unique(self.unique_minerals),np.unique(mask_crop))))
        self.unique_minerals_count = len(self.unique_minerals)

    def extract_colors(self, ref_images = False) -> tuple:
        '''
        Extract mineral and background colors from input and reference images.
                Returns:
                        color_spaces (tuple): tuple of ndarrays with shape (N_minerals + 1, 3)
                        (
                        inp_color_space (ndarray),
                        ref_color_space (ndarray)
                        ) 
        '''
        self.get_unique_minerals_count()

        real_size = 0
        indexes = []
        ref_colors = []
        inp_colors = []
        lower_bound = 0 if ref_images else 1
        for i in range(len(self.inp_crops)):
            ref_crop = self.ref_crops[i]
            inp_crop = self.inp_crops[i]
            mask_crop = self.mask_crops[i]
            h, w, _ = ref_crop.shape
            threshold_data = h * w / 10
            for j in range(lower_bound, self.unique_minerals_count):
                mineral = self.unique_minerals[j]
                ref_img_temp = ref_crop.copy()
                data_ref = ref_img_temp[mask_crop == mineral].reshape(-1, 3)
                if (data_ref.shape[0] > threshold_data):
                    real_size += 1
                    indexes += [j]    

                    ref_color = stats.mode(data_ref).mode[0]
                    ref_img_temp[:] = ref_color

                    inp_img_temp = inp_crop.copy()
                    inp_data = inp_img_temp[mask_crop == mineral].reshape(-1, 3)
                    inp_color = stats.mode(inp_data).mode[0]
                    inp_img_temp[:] = inp_color

                    ref_colors += [ref_color]
                    inp_colors += [inp_color]

                    ref_img_for_plot = ref_crop.copy()
                    ref_img_for_plot[mask_crop != mineral] = [0,0,0]
                    inp_imf_for_plot = inp_crop.copy()
                    inp_imf_for_plot[mask_crop != mineral] = [0,0,0]
                    f = plt.subplots(2,2, figsize=(15,15))[0]
                    f.axes[0].imshow(ref_img_for_plot)
                    f.axes[1].imshow(ref_img_temp)
                    f.axes[2].imshow(inp_imf_for_plot)
                    f.axes[3].imshow(inp_img_temp)
                    f.savefig(f'{self.dir_path}/img{self.index}_mineral{j}_extraction.png')

                
        # inp_bg_color, ref_bg_color = self.extract_background_color()

        # ref_colors += [ref_bg_color]
        # inp_colors += [inp_bg_color]

        ref_colors = np.array(ref_colors)
        inp_colors = np.array(inp_colors)

        f = plt.subplots((math.ceil(np.sqrt(real_size))),
                         math.ceil(np.sqrt(real_size)), 
                         figsize=(15,15))[0]
        for i in range(real_size):
            image = np.zeros((300, 300, 3), np.uint8)
            image[:] = ref_colors[i]
            f.axes[i].imshow(image)
        f.savefig(f'{self.dir_path}/img{self.index}_ref_image_colors_extracted.png')
        for i in range(real_size):
            image = np.zeros((300, 300, 3), np.uint8)
            image[:] = inp_colors[i]
            f.axes[i].imshow(image)
        f.savefig(f'{self.dir_path}/img{self.index}_inp_image_colors_extracted.png')

        return (inp_colors, ref_colors)
    
    def extract_colors_from_image(self, image, mask, index, ref_images = False) -> tuple:
        '''
        Extract mineral and background colors from given image.
                Returns:
                        color_space (ndarray),
        '''
        real_size = 13
        colors = np.empty((13, 1, 3))
        colors[:] = np.nan
        h, w, _ = image.shape
        threshold_data = h * w / 10
        lower_bound = 0 if ref_images else 1
        for j in range(lower_bound, real_size):
            mineral = j
            ref_img_temp = image.copy()
            data_ref = ref_img_temp[mask == mineral].reshape(-1, 3)
            if (data_ref.shape[0] > threshold_data):

                ref_color = np.mean(data_ref, axis=0)
                ref_img_temp[:] = ref_color

                colors[j] = ref_color

                ref_img_for_plot = image.copy()
                ref_img_for_plot[mask != mineral] = [0,0,0]

                f = plt.subplots(2,2, figsize=(15,15))[0]
                f.axes[0].imshow(ref_img_for_plot)
                f.axes[1].imshow(ref_img_temp)

                f.savefig(f'{self.dir_path}/img{index}_mineral{j}_extraction.png')

        f = plt.subplots((math.ceil(np.sqrt(real_size))),
                         math.ceil(np.sqrt(real_size)), 
                         figsize=(15,15))[0]
        for i in range(real_size):
            image = np.zeros((300, 300, 3), np.uint8)
            image[:] = colors[i]
            f.axes[i].imshow(image)
        f.savefig(f'{self.dir_path}/img{index}_colors_extracted.png')

        return colors

    def extract_background_color(self) -> tuple:
        '''
        Extract background colors from input and reference images.
                Returns:
                        background_colors (tuple): tuple of ndarrays with shape (3, )
                        (
                        inp_bg_color (ndarray),
                        ref_bg_color (ndarray)
                        ) 
        '''
        clt_ref = KMeans(n_clusters=1)
        ref_img_temp = self.ref_image.copy()
        clt_ref.fit(ref_img_temp.reshape(-1, 3))
        ref_bg_color = clt_ref.cluster_centers_[0]
        ref_img_temp[:] = ref_bg_color

        clt_inp = KMeans(n_clusters=1)
        inp_img_temp = self.input_image.copy()
        clt_inp.fit(inp_img_temp.reshape(-1, 3))
        inp_bg_color = clt_inp.cluster_centers_[0]
        inp_img_temp[:] = inp_bg_color


        f = plt.subplots(2,2, figsize=(15,15))[0]
        f.axes[0].imshow(self.ref_image)
        f.axes[1].imshow(ref_img_temp)
        f.axes[2].imshow(self.input_image)
        f.axes[3].imshow(inp_img_temp)
        f.savefig(f'{self.dir_path}/img{self.index}_background_color_extraction.png')

        return (inp_bg_color, ref_bg_color)