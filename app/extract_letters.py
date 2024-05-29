import cv2
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps


def split_letters(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path, 0)

    # Бинаризация изображения
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Поиск контуров
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортировка контуров по положению левого верхнего угла
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    letter_images = []

    # Создание нового изображения для каждого контура
    for i, ctr in enumerate(contours):
        # Получение координат и размеров контура
        x, y, w, h = cv2.boundingRect(ctr)

        # Извлечение буквы
        roi = image[y:y+h, x:x+w]

        # Инвертирование цвета буквы
        roi = cv2.bitwise_not(roi)

        # Добавление белого фона, чтобы изображение стало квадратным
        square_size = max(w, h)
        square_roi = np.full((square_size, square_size), 255, dtype=np.uint8)
        x_offset = (square_size - w) // 2
        y_offset = (square_size - h) // 2
        square_roi[y_offset:y_offset+h, x_offset:x_offset+w] = roi

        # Конвертация изображения в формат PIL
        pil_image = Image.fromarray(square_roi)

        # Добавление белых отступов по краям изображения
        border_size = 30  # Размер отступа в пикселях
        expanded_image = ImageOps.expand(pil_image, border=border_size, fill='white')
        pil_image = expanded_image

        # Сохранение изображения в BytesIO
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        letter_images.append(img_bytes)

    return letter_images


if __name__ == "__main__":
    # Пример использования функции
    letter_images = split_letters('images/letters.png')
    for i, img_bytes in enumerate(letter_images):
        with open('letter_{}.png'.format(i), 'wb') as f:
            f.write(img_bytes.read())

