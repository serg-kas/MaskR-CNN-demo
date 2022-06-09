"""
Демонстрация предобученной архитектуры Mask-R-CNN в
нескольких режимах работы (object detection, segmentation,remove_background).
При запуске обрабатывает все файлы из папки source_files
Результат помещается в папку out_files.
Режим работы можно передать параметром командной строки.
Достаточно передать подстроку из названия режима.
"""
# Модуль с функциями
import run
# Прочее
import sys
import os

# Допустимые форматы изображений
img_type_list = ['.jpg', '.jpeg', '.png']
# Режимы работы
operation_mode_list = ['object_detection', 'instant_segmentation',
                       'remove_background', 'remove_background_blur',
                       'remove_background_opencv_canny', 'remove_background_test']
default_mode = operation_mode_list[-1]  # режим работы по умолчанию
# Модель URL
MODEL_URL = "https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1"


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
    # out_files = sorted(os.listdir(out_path))
    img_files = []
    for f in source_files:
        filename, file_extension = os.path.splitext(f)
        # if not (('out_'+f) in out_files):
        if file_extension in img_type_list:
            img_files.append(f)

    # Если обрабатывать нечего, то выходим
    if len(img_files) == 0:
        print('Нет картинок для обработки.')
        return
    else:
        print('Картинок для обработки: {0}'.format(len(img_files)))

    # Парсинг режима работы
    # Например по параметру командной строки obj установит режим работы object_detection и т.п.
    for mode in operation_mode_list:
        if operation_mode in mode:
            operation_mode = mode
            break
    print('Режим работы: {}'.format(operation_mode))

    # Получаем модель и обрабатываем картинки
    if 'opencv' not in operation_mode:
        print('Загружаем модель...')
        model = run.get_model(MODEL_URL)

    for file in img_files:
        # полные пути к файлам
        img_file = os.path.join(source_path, file)
        out_file = os.path.join(out_path, file)
        # Вызов функции предикта
        if operation_mode == 'object_detection':
            run.img_obj_detection(model, img_file, out_file)
        if operation_mode == 'instant_segmentation':
            run.img_inst_segmention(model, img_file, out_file)
        if operation_mode == 'remove_background':
            run.img_rem_background_blur(model, img_file, out_file, cont_blur=False)
        if operation_mode == 'remove_background_blur':
            run.img_rem_background_blur(model, img_file, out_file, cont_blur=True)
        if operation_mode == 'remove_background_opencv_canny':
            run.img_rem_background_opencv_canny(img_file, out_file)
        if operation_mode == 'remove_background_test':
            run.img_rem_background_test(model, img_file, out_file)


if __name__ == '__main__':
    operation_mode = default_mode if len(sys.argv) <= 1 else sys.argv[1]
    source_path = 'source_files' if len(sys.argv) <= 2 else sys.argv[2]
    out_path = 'out_files' if len(sys.argv) <= 3 else sys.argv[3]
    #
    process(operation_mode, source_path, out_path)
