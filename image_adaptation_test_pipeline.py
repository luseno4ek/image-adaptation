from template_matching import *
from extracting_color_spaces import *
from image_adaptation import *
from color_calibration import *
import cv2 as cv
import os
import numpy as np
import datetime

#################### ПОДГОТОВКА К РАБОТЕ СКРИПТА ####################
# чтение изображений
# создание директорий

# Директория с искаженными изображениями
input_images_directory = 'imgs'
# Директория с масками
masks_directory = 'masks'
# Директория с референсными изображениями
ref_images_directory = 'ref'

# Сюда будем считывать изображения из директорий
input_images = []
ref_images = []
masks = []

# Читаем изображения и складываем в списки
for f in os.listdir(input_images_directory):
    new_img = cv.imdecode(np.fromfile(input_images_directory + '\\' + f, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    input_images.append(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))

for f in os.listdir(ref_images_directory):
    new_img = cv.imdecode(np.fromfile(ref_images_directory + '\\' + f, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    ref_images.append(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))

for f in os.listdir(masks_directory):
    masks.append(cv.imdecode(np.fromfile(masks_directory + '\\' + f, dtype=np.uint8), cv.IMREAD_UNCHANGED))

# Проверка тестовых данных
if (len(input_images) != len(masks)) or (len(ref_images) == 1):
    print('!!!ERROR!!!: Input error. Check that len(input_images) == len(masks) and len(ref_images) == 1')
    exit()

# Графики и результаты адаптации будут храниться по этому пути
main_path = (input_images_directory.split('\\')[-1] + '_'
            f'{datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%Y")}')

# Подпапки для графиков и результатов
result_dir_path = (f'results')
dir_path = (f'graphics')
result_dir_path = f'{main_path}\\{result_dir_path}'
dir_path = f'{main_path}\\{dir_path}'

# Создаем папки
os.mkdir(main_path)
os.mkdir(dir_path)
os.mkdir(result_dir_path)

#################### ОСНОВНОЙ СКРИПТ ####################

# Создаем экземпляр класса ImageAdaptator
image_adaptator = ImageAdaptator(1, dir_path, result_dir_path, main_path)

# Получаем референсное цветовое пространство c референсного изображения
ref_cs_dict = image_adaptator.GetRefColorSpace([ref_images[0]], [masks[0]], "ref_cs_test_pipeline")

# Адаптируем каждое входное изображение по референсному цветовому пространству
size = len(input_images)

for i in range(0, size):
    image_adaptator.CalibrateWithoutRefImg(input_images[i], masks[i], ref_cs_dict)
    image_adaptator.index += 1

