# Адаптация изображений аншлифов

## Работа с классом ImageAdaptator

Работа с моделью осуществляется через класс [ImageAdaptator](https://github.com/luseno4ek/image-adaptation/blob/main/image_adaptation.py).

Методы класса позволяют:
-  получить референсное и/или искаженное цветовые пространства;
-  адаптировать изображение под референсное изображение;
-  адаптировать изображение под референсное цветовое пространство;
-  применить уже имеющуюся матрицу цветовой коррекции к изображению.

В своих скриптах можно использовать публичные методы класса. Для каждого публичного метода есть докстринга с краткой информацией о методе:
```
def GetRefColorSpace(self, images, masks, name) -> Dict:
    '''
    Gets colorspace dict from given images with masks.

    Parameters:
        images (list of ndarrays): input images
        masks  (list of ndarrays): masks for input images
        name                (str): name of experiment, used to save colorspace image file
    '''
```

## Тестовый пайплайн

Тестовый пайплайн работы с классом ImageAdaptator представлен в файле [image_adaptation_test_pipeline](https://github.com/luseno4ek/image-adaptation/blob/main/image_adaptation_test_pipeline.py).

Логика работы тестового пайплайна:
- Входные данные - 8 искаженных изображений, 8 масок для искаженных изображений, 1 референсное изображение, совмещенное с маской №1
- Извлекаем из референсного изображения референсное цветовое пространство
- Адаптируем искаженные изображения под референсное цветовое пространство

После отработки скрипт создает папку с результатами и метаданными в виде графиков выделенных минералов, цветовых пространств идр.


## Скрипты, использованные в дипломе

Примеры использования ImageAdaptator в работе над дипломом представлены в файле [image_adaptation_script](https://github.com/luseno4ek/image-adaptation/blob/main/image_adaptation_script.py).
