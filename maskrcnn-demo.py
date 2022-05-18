"""
Демонстрация предобученной архитектуры Mask-R-CNN в
нескольких режимах работы (object detection, segmentation,
remove_background и т.д.)
При запуске обрабатывает все файлы из папки source_files
Результат помещает в папку out_files, добавляя к имени "out_"
Если файл с таким названием уже обрабатывался, то его не трогает.
Видео во время обработки отображается.
"""
# Модуль с функциями
import run
# Прочее
import os
import sys
import tensorflow_hub as hub
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Допустимые форматы
img_type_list = ['.jpg', '.jpeg', '.png']

# Режимы работы
# TODO: режимы работы пока не задействованы
operation_mode_list = ['object_detection', 'instant_segmentation', 'remove_background']


def process(operation_mode, source_path, out_path):
    """
    @param: operation_mode режим работы
    @param: source_path путь к каталогу с файлами
    @param: out_path путь результатам
    """
    # Создадим папки для файлов, если их нет
    if not (source_path in os.listdir('.')):
        os.mkdir(source_path)
    if not (out_path in os.listdir('.')):
        os.mkdir(out_path)

    # Создадим список файлов картинок для обработки
    source_files = sorted(os.listdir(source_path))
    out_files = sorted(os.listdir(out_path))
    img_files = []
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # print(f,filename,file_extension)
        if not (('out_'+f) in out_files):
            if file_extension in img_type_list:
                img_files.append(f)

    # Закомментировать, если не нужно скрывать наличие GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    # Получаем модель
    model_name = "https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"
    model = hub.load(model_name)

    # Обрабатываем картинки
    for img in img_files:
        # полные пути к файлам
        img_file = os.path.join(source_path, img)
        out_file = os.path.join(out_path, 'out_' + img)
        # Вызов функции предикта
        if operation_mode == 'object_detection':
            _ = run.img_detection(model, img_file, out_file)
        if operation_mode == 'instant_segmentation':
            _ = run.img_segmention(model, img_file, out_file)
        if operation_mode == 'remove_background':
            _ = run.img_background(model, img_file, out_file)

    # Сообщаем что обработали
    if len(img_files) == 0:
        print('Нет картинок для обработки.')
    else:
        print('Обработали картинок: {0}'.format(len(img_files)))


if __name__ == '__main__':
    operation_mode = operation_mode_list[2] if len(sys.argv) <= 1 else sys.argv[1]
    source_path = 'source_files' if len(sys.argv) <= 2 else sys.argv[2]
    out_path = 'out_files' if len(sys.argv) <= 3 else sys.argv[3]
    #
    process(operation_mode, source_path, out_path)
