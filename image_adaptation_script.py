from template_matching import *
from extracting_color_spaces import *
from image_adaptation import *
from color_calibration import *
import cv2 as cv
import os
import numpy as np
import datetime
import pickle

input_images_directory = 'C:\\Users\\olind\\Documents\\Курсач\\Мои тесты\\частичная разметка\\5+6\\imgs'
masks_directory = 'C:\\Users\\olind\\Documents\\Курсач\\Мои тесты\\частичная разметка\\5+6\\masks'

ref_images_directory = 'C:\\Users\\olind\\Documents\\Курсач\\Мои тесты\\частичная разметка\\5+6\\ref'
ref_cs_directory = 'C:\\Users\\olind\\Documents\\Курсач\\Мои тесты\\cs\\colorspace.pkl'

input_images = []
ref_images = []
masks = []

for f in os.listdir(input_images_directory):
    new_img = cv.imdecode(np.fromfile(input_images_directory + '\\' + f, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    input_images.append(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))

for f in os.listdir(ref_images_directory):
    new_img = cv.imdecode(np.fromfile(ref_images_directory + '\\' + f, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    ref_images.append(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))

for f in os.listdir(masks_directory):
    masks.append(cv.imdecode(np.fromfile(masks_directory + '\\' + f, dtype=np.uint8), cv.IMREAD_UNCHANGED))

main_path = (input_images_directory.split('\\')[-1] + '_'
            f'{datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%Y")}')

size = len(masks)
result_dir_path = (f'results')
dir_path = (f'graphics')

result_dir_path = f'{main_path}\\{result_dir_path}'

dir_path = f'{main_path}\\{dir_path}'

os.mkdir(main_path)
os.mkdir(dir_path)
os.mkdir(result_dir_path)


ccms = np.zeros((len(masks), 4, 3))

ref_cs_from_dict = {}

with open(ref_cs_directory, 'rb') as f:
    ref_cs_from_dict = pickle.load(f)


#################### Задача "дообучения" модели" ####################
#   Известны несколько масок изображений,
#   последовательно обучаем модель на размеченных изображениях, 
#   в конце должны получить усредненную матрицу цветовой коррекции.
#   Ниже представлены два решения:
#   1. Последовательно обучаем модель на каждом из изображений, 
#   после усредняем полученные матрицы с некоторыми коэффициентами
#   2. Из каждого изображения вытягиваем цветовые пространства,
#   усредняем цветовые пространства и учим модель на усредненном цветовом пространстве


# СПОСОБ 1: БЕРЕМ СРЕДНЕЕ ПО ВЫУЧЕННЫМ МАТРИЦАМ ЦВЕТОКОРРЕКЦИИ
image_adaptator = ImageAdaptator(0, dir_path, result_dir_path, main_path)
for i in range(len(masks)):
    ccm = image_adaptator.CalibrateAlone(input_images[i], ref_images[i], masks[i])
    ccms[i] = ccm
    image_adaptator.index += 1

result_ccm = image_adaptator.GetResultCCM(masks, ccms)
print(f'\n||RESULT||: result ccm = \n {result_ccm}')
print('===============================================')

# СПОСОБ 2: БЕРЕМ СРЕДНЕЕ ОТ ЦВЕТОВЫХ ПРОСТРАНСТВ
input_dicts_list = []
image_adaptator = ImageAdaptator(0, dir_path, result_dir_path, main_path)
for i in range(len(masks)):
    inp_colors = image_adaptator.GetColorSpace(input_images[i], masks[i], ref_cs_from_dict)
    input_dicts_list += [inp_colors]
    image_adaptator.index += 1
inp_colors, ref_colors = image_adaptator.GetRefAndInputCS(ref_cs_from_dict, input_dicts_list)
print(f'\n||RESULT||: input cs = \n {inp_colors}')
print(f'\n||RESULT||: ref cs = \n {ref_colors}')
result_ccm = image_adaptator.CalibrateWithColorSpaces(input_images[len(masks)-1], inp_colors, ref_colors)
print(f'\n||RESULT||: result ccm = \n {result_ccm}')
print('===============================================')

# Адаптируем изображения, для которых не было масок, чтобы измерить качество
for i in range(len(masks), len(input_images)):
    image_adaptator.index = i
    ccm = image_adaptator.InferImage(input_images[i], result_ccm)




#################### Задача "калибровки" модели" ####################
#   Известны несколько масок изображений,
#   изображения, для которых маски известны, адаптируем поодиночке, 
#   изображения, для которых нет масок, адаптируем, используя одну из 
#   полученных матриц цветовой коррекции.

CALIBRATE_ALONE : int = size
GET_MASK_FROM: int = 0
ccms = []

for i in range(0,CALIBRATE_ALONE):
    print('===============================================')
    print(f'\n||INFO||: Template matching started {i+1}/{size}')
    input_image = input_images[i]
    ref_image = ref_images[i]
    ref_mask = masks[i]

    template_matcher = TemplateMatcher(input_image, ref_image, ref_mask, i + 1, dir_path)
    res_matcher = template_matcher.get_matched_templates()
    if (res_matcher == None):
        continue
    
    inp_crops, ref_crops, mask_crops = [input_image], [ref_image],[ref_mask[:,:,0]]       
    
    print(f'||INFO||: Unique minerals in mask {np.unique(ref_mask)}\n')
    
    start_time_alone = datetime.datetime.now()

    print(f'\n||INFO||: Extracting colorspaces started {i+1}/{size}')
    colors_extractor = ColorSpacesExtractor(input_image, ref_image, 
                                            inp_crops, ref_crops, mask_crops, i + 1, dir_path)
    inp_colors, ref_colors = colors_extractor.extract_colors()
    print(f'||INFO||: Extracting colorspaces finished {i+1}/{size}\n')
    print(f'\n||INFO||: Color calibration started {i+1}/{size}')
    colors_calibrator = ColorCalibrator(input_image, 
                                        inp_colors, ref_colors, i + 1, 
                                        result_dir_path,
                                        main_path,
                                        ccm_shape='4x3',
                                        distance='de00',
                                        gamma = 2,
                                        linear = 'gamma')
    (result_image, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)
    ccms.append(ccm)
    print('||STATISTICS||: Working time alone: ', datetime.datetime.now() - start_time_alone);
    print(f'||INFO||: ColorCalibration finished {i+1}/{size}\n')
    print('===============================================')

last_ccm = None


if GET_MASK_FROM == 0:
    last_ccm = ccms[-1]
    start_time = datetime.datetime.now()
else:
    print('===============================================')
    print(f'\n||INFO||: Getting reference CCM from img {GET_MASK_FROM}\n')
    print('===============================================')
    print(f'\n||INFO||: Template matching started')
    input_image = input_images[GET_MASK_FROM-1]
    ref_image = ref_images[GET_MASK_FROM-1]
    ref_mask = masks[GET_MASK_FROM-1]

    template_matcher = TemplateMatcher(input_image, ref_image, ref_mask, GET_MASK_FROM, dir_path)
    res_matcher = template_matcher.get_matched_templates()
    
    inp_crops, ref_crops, mask_crops = res_matcher        

    print(f'||INFO||: Template matching finished \n')

    start_time = datetime.datetime.now()

    print(f'\n||INFO||: Extracting colorspaces started')
    colors_extractor = ColorSpacesExtractor(input_image, ref_image, 
                                            inp_crops, ref_crops, mask_crops, 
                                            GET_MASK_FROM, dir_path)
    inp_colors, ref_colors = colors_extractor.extract_colors()
    print(f'||INFO||: Extracting colorspaces finished \n')
    print(f'\n||INFO||: Color calibration started')
    colors_calibrator = ColorCalibrator(input_image, 
                                        inp_colors, ref_colors, GET_MASK_FROM, 
                                        result_dir_path,
                                        main_path,
                                        ccm_shape='4x3',
                                        distance='de00',
                                        gamma = 2,
                                        linear = 'gamma')
    (result_image, ccm) = colors_calibrator.get_calibrated_image(return_ccm=True)
    last_ccm = ccm
    print(f'||INFO||: ColorCalibration finished \n')
    print('===============================================')



for i in range(CALIBRATE_ALONE, size):
    print('===============================================')
    print(f'\n||INFO||: Applying color correction matrix {i+1}/{size}')
    input_image = input_images[i]
    ref_image = ref_images[i]
    ref_mask = masks[i]

    start_time_apply = datetime.datetime.now()

    colors_calibrator = ColorCalibrator(input_image, 
                                        None, None, i + 1, 
                                        result_dir_path,
                                        main_path,
                                        only_infer=True)

    result_image = colors_calibrator.apply_calibrating_mask(last_ccm)
    
    print('||STATISTICS||: Working time apply: ', datetime.datetime.now() - start_time_apply)


    print(f'||INFO||: Applying CCM finished {i+1}/{size}\n')
    print('===============================================')

print('||INFO||: Image adaptation finished\n')

print('||STATISTICS||: Working time: ', datetime.datetime.now() - start_time);



