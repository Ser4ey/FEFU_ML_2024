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
    start_text = "Привет! Это бот для распознавания рукописных русских букв!" \
           "\n\nGithub репозиторий проекта: https://github.com/Ser4ey/FEFU_ML_2024.git"

    bot.send_message(message.chat.id, start_text)


@bot.message_handler(commands=['help'])
def handle_help(message):
    help_text = "Отправьте боту изображение с русской буквой. Если на изображении несколько букв, добавьте подпись '2'"

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

    # сохраняем фото
    path_to_file = f'download/{file_info.file_id}.jpg'
    with open(path_to_file, 'wb') as file:
        file.write(downloaded_file)

    # подпись к фото
    caption = message.caption
    if caption == "2":
        answer_to_user = get_many_letter_answer(path_to_file)
        bot.send_message(message.chat.id, answer_to_user)
    else:
        answer_to_user = get_one_letter_answer(path_to_file)
        bot.send_message(message.chat.id, answer_to_user)

        buf = model.get_img(img_file)
        bot.send_photo(message.chat.id, buf)


def get_many_letter_answer(path_to_img) -> str:
    images = split_letters(path_to_img)

    answer_to_user = ""
    for i in range(len(images)):
        res = model.predict_img(images[i])
        letter = list(res.keys())[0]
        answer_to_user += letter

    print(answer_to_user)
    return answer_to_user


def get_one_letter_answer(path_to_img) -> str:
    res = model.predict_img(path_to_img)
    print(res)

    count = 0
    answer_to_user = ""
    for k, v in res.items():
        answer_to_user += f"{k}: {v:.3f}%\n"
        count += 1
        if count >= 5:
            break
    return answer_to_user


bot.infinity_polling()
