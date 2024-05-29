import os
import telebot
from io import BytesIO

from model import MLModel
from extract_letters import split_letters

# Получение значения переменной окружения
model = MLModel(os.getenv('PATH_TO_MODEL'))
bot = telebot.TeleBot(os.getenv('BOT_TOKEN'))


@bot.message_handler(commands=['start'])
def handle_start(message):
    start_text = "Привет! Это бот для распознования рукописных русских букв!" \
           "\n\nGithub репозиторий проекта: https://github.com/Ser4ey/FEFU_ML_2024.git"

    bot.send_message(message.chat.id, start_text)


@bot.message_handler(commands=['help'])
def handle_help(message):
    help_text = "Отправте боту изображение с рксской буквой. Если на изображении несколько букв, добавьте подпись '2'"

    bot.send_message(message.chat.id, help_text)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.send_message(message.chat.id, "Обработка...")

    # Получаем самое большое доступное фото
    photo = max(message.photo, key=lambda x: x.file_size)
    # Скачиваем фото
    file_info = bot.get_file(photo.file_id)
    print(file_info)
    downloaded_file = bot.download_file(file_info.file_path)

    img_file = BytesIO(downloaded_file)
    img_file.seek(0)

    # подпись к фото
    caption = message.caption
    if caption == "2":
        path_to_file = f'download/{file_info.file_id}.jpg'
        with open(path_to_file, 'wb') as new_file:
            new_file.write(downloaded_file)

        answer_text = get_text_by_images(path_to_file)
        bot.send_message(message.chat.id, answer_text)
        return

    # # Сохраняем фото в файл
    # path_to_file = f'download/{file_info.file_id}.jpg'
    # with open(path_to_file, 'wb') as new_file:
    #     new_file.write(downloaded_file)

    # res = model.predict_img(path_to_file)
    res = model.predict_img(img_file)
    print(res)

    count = 0
    ans = ""
    for k, v in res.items():
        ans += f"{k}: {v:.3f}%\n"
        count += 1
        if count >= 5:
            break
    bot.send_message(message.chat.id, ans)

    buf = model.get_img(img_file)
    bot.send_photo(message.chat.id, buf)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


def get_text_by_images(path_to_img):
    images = split_letters(path_to_img)
    text = ""
    for i in range(len(images)):
        print(i)
        res = model.predict_img(images[i])
        letter = list(res.keys())[0]
        text += letter
        print(res)
        print("\n")

    print(text)
    return text


bot.infinity_polling()


