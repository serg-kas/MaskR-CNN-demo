"""
Демонстрация предобученной архитектуры Mask-R-CNN.
При запуске обрабатывает все файлы из папки source_files
Результат помещает в папку out_files, добавляя к имени "out_"
Если файл с таким названием уже обрабатывался, то его не трогает.
Видео во время обработки отображается.
"""
# Модуль с функцями
import utils
# Прочее
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Допустимые форматы
img_type_list = ['.jpg', '.jpeg', '.png']
vid_type_list = ['.mp4', '.avi']


def process(source_path, out_path, model_path):
    """
    @param: source_path путь к каталогу с файлами
    @param: out_path путь результатам
    @param: model_path Путь к модели
    """
    # Создадим папки для файлов, если их нет
    if not (source_path in os.listdir('.')):
        os.mkdir(source_path)
    if not (out_path in os.listdir('.')):
        os.mkdir(out_path)

    # В папке должен быть файл модели
    # assert model_path in os.listdir('.'), 'В папке программы должен быть файл модели'

    # Создадим список файлов для обработки
    source_files = sorted(os.listdir(source_path))
    out_files = sorted(os.listdir(out_path))
    # Раздельные списки для картинок и видео
    img_files = []
    vid_files = []
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # print(f,filename,file_extension)
        if not (('out_'+f) in out_files):
            if file_extension in img_type_list:
                img_files.append(f)
            if file_extension in vid_type_list:
                vid_files.append(f)

    # Получаем модель
    # model = cigadet.get_model(model_PATH, '')
    # model = utils.get_model(model_path, '/cpu:0')

    # Обрабатываем картинки
    for img in img_files:
        # полные пути к файлам
        img_file = os.path.join(source_path, img)
        out_file = os.path.join(out_path, 'out_' + img)
        # Вызов функции предикта
        # _ = utils.img_predict(model, img_file, out_file)

    # Обрабатываем видео
    for vid in vid_files:
        # полные пути к файлам
        vid_file = os.path.join(source_path, vid)
        out_file = out_file = os.path.join(out_path, 'out_' + vid)
        # Вызов функции предикта
        # _ = utils.vid_predict(model, vid_file, out_file)

    # Сообщаем что обработали
    if len(img_files) == 0:
        print('Нет картинок для обработки.')
    else:
        print('Обработали {0} картинок.'.format(len(img_files)))
    if len(vid_files) == 0:
        print('Нет видео для обработки.')
    else:
        print('Обработали {0} видео.'.format(len(vid_files)))


if __name__ == '__main__':
    source_path = 'source_files' if len(sys.argv) <= 1 else sys.argv[1]
    out_path = 'out_files' if len(sys.argv) <= 2 else sys.argv[2]
    model_path = 'best-4-model-inception.h5' if len(sys.argv) <= 3 else sys.argv[3]

    process(source_path, out_path, model_path)
