import io

import tensorflow as tf
import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np



class MLModel:
    class_names = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О',
                   'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

    def __init__(self, path_to_model: str):
        self.model = load_model(path_to_model)

    def show_img(self, path_to_img: str, do_rescale=True):
        image = keras.utils.load_img(
            path_to_img,
            target_size=(28, 28),
            color_mode="grayscale"
        )

        # нормадизуем изображение
        if do_rescale:
            rescale = layers.Rescaling(1. / 255)
            image = rescale(image)

        image_array = img_to_array(image)
        plt.imshow(image_array.astype('uint8'))
        plt.show()

    def get_img(self, path_to_img: str, do_rescale=True):
        image = keras.utils.load_img(
            path_to_img,
            target_size=(28, 28),
            color_mode="grayscale"
        )
        image = tf.keras.preprocessing.image.img_to_array(image)

        # Нормализуем изображение
        if do_rescale:
            image = image / 255.0
        # print(image)

        # Конвертируем тензор в PIL Image
        image = tf.keras.preprocessing.image.array_to_img(image)

        # Конвертируем PIL Image в байты
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        return img_bytes
        # image_array = img_to_array(image)
        #
        # # Преобразуем массив NumPy в изображение и сохраняем его в буфер
        # image = Image.fromarray(np.uint8(image_array * 255))
        # buf = BytesIO()
        # image.save(buf, format='JPEG')
        # buf.seek(0)
        #
        # return buf


    def predict_img(self, path_to_img: str):
        # self.show_img(path_to_img, False)
        image = keras.utils.load_img(
            path_to_img,
            target_size=(28, 28),
            color_mode="grayscale"
        )

        # нормадизуем изображение
        rescale = layers.Rescaling(1. / 255)
        image = rescale(image)

        # преобразуем изображение для подачи нейронке на вход
        input_arr = keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.

        res = self.model.predict(input_arr)
        # print(res)
        print(np.argmax(res))
        print(MLModel.class_names[np.argmax(res)])

        # Получение индексов отсортированных по вероятности классов
        sorted_indices = np.argsort(res[0])[::-1]
        print(sorted_indices)

        # Вывод трех самых вероятных результатов и их процентных вероятностей
        for i in range(3):
            index = sorted_indices[i]
            print(f"Class: {MLModel.class_names[index]}, Probability: {res[0][index] * 100:.2f}%")

        ans = {}
        for i in sorted_indices:
            ans[MLModel.class_names[i]] = res[0][i] * 100
        return ans


if __name__ == "__main__":
    m = MLModel("v8.h5")

    r = m.predict_img("4.jpg")
    print(r)
