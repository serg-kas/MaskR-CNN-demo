"""
Модуль с функциями
"""
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # закомментировать для использования GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # уровень 2 - только сообщения об ошибках
import tensorflow as tf
import tensorflow_hub as hub
#
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Размер к которому приводить изображение (только в функциях удаления фона)
IMG_SIZE = 512


# Функция получения модели
def get_model(model_url):
    """
    :param model_name: путь к  модели для загрузки
    :return: model: загруженная модель
    """
    # if tf.test.is_gpu_available():
    #     print('GPU found')
    # else:
    #     print('No GPU found')

    gpu_present = bool(len(tf.config.list_physical_devices('GPU')))
    if gpu_present:
        print('GPU found')
    else:
        print('No GPU found')

    time_start = time.time()
    model = hub.load(model_url)
    time_end = time.time() - time_start
    print('Время загрузки модели: {0:.1f}'.format(time_end))
    return model


# Функция наложения маски
def apply_mask(image, mask):
    """
    :param image: исходное изображение
    :param mask: маска
    :return: изображение с наложенной маской
    """
    image[:, :, 0] = np.where(mask == 0, 127, image[:, :, 0])
    image[:, :, 1] = np.where(mask == 0, 127, image[:, :, 1])
    image[:, :, 2] = np.where(mask == 0, 127, image[:, :, 2])
    return image


# Функция наложения оригинального изображения везде кроме маски
# def apply_inst(image_orig, image_out, mask):
#     image_out[:, :, 0] = np.where(mask != 0, image_orig[:, :, 0], 127)
#     image_out[:, :, 1] = np.where(mask != 0, image_orig[:, :, 1], 127)
#     image_out[:, :, 2] = np.where(mask != 0, image_orig[:, :, 2], 127)
#     return image_out


# Функция ресайза картинки через opencv
def img_resize_cv(image, img_size):
    """
    :param image: исходное изображение
    :param img_size: размер к которому приводить изображение
    :return: изображение после ресайза
    """
    curr_w = image.shape[1]
    curr_h = image.shape[0]
    # Рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_img = img_size / curr_w
    else:
        scale_img = img_size / curr_h
    # Новые размеры изображения
    new_width = int(curr_w * scale_img)
    new_height = int(curr_h * scale_img)
    # делаем ресайз к целевым размерам
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image


# Функция размытия контура маски
def cut_and_blur_contour_cv(img, mask, cnt_thickness=4, kernel=(5, 5)):
    """
    Idea from https://ru.stackoverflow.com/questions/950969
    :param img:
    :param mask:
    :param cnt_thickness:
    :param kernel:
    :return: result: изображение с нанесенной маской и размытым контуром
    """
    # img = cv2.bitwise_and(img, img, mask=mask)
    img = apply_mask(img, mask)  # своя функция наложения маски

    tmp = img.copy()
    # prepare a blurred image
    blur = cv2.GaussianBlur(img, kernel, 0)

    # find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours using passed [cnt_thickness] on a temporary image
    _ = cv2.drawContours(tmp, contours, 0, (0, 255, 0), cnt_thickness)

    # create contour mask
    hsv = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

    # apply contour mask
    tmp = cv2.bitwise_and(blur, blur, mask=mask)

    result = np.where(tmp > 0, blur, img)
    # Image.fromarray(result).show()
    return result


# Функция предикта object detection
def img_obj_detection(model, img_file, out_file):
    """
    Использован код из Tensorflow Object detection API
    :param model: ранее загруженная модель
    :param img_file: путь к исходному файлу картинки
    :param out_file: путь куда записывать готовый файл
    """

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # запускаем предикт
    time_start = time.time()
    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())
    time_end = time.time() - time_start
    print('Время предикта: {0:.1f}'.format(time_end))

    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    result = Image.fromarray(image_np_with_detections[0])
    result.save(out_file)
    return


# Функция предикта instance segmentation
def img_inst_segmention(model, img_file, out_file):
    """
    Использован код из Tensorflow Object detection API
    :param model: ранее загруженная модель
    :param img_file: путь к исходному файлу картинки
    :param out_file: путь куда записывать готовый файл
    """

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # Засекаем время и запускаем предикт
    time_start = time.time()
    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())
    time_end = time.time() - time_start
    print('Время предикта: {0:.1f}'.format(time_end))

    label_id_offset = 0
    image_np_with_mask = image_np.copy()

    if 'detection_masks' in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
        detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes,
            image_np.shape[1], image_np.shape[2])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        result['detection_masks_reframed'] = detection_masks_reframed.numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_mask[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        instance_masks=result.get('detection_masks_reframed', None),
        line_thickness=8)

    result = Image.fromarray(image_np_with_mask[0])
    result.save(out_file)
    return


# Функция удаления фона с возможностью размывания контура сегментации
def img_rem_background_blur(model, img_file, out_file, cont_blur=False):
    """
    Использован код из Tensorflow Object detection API и библиотека opencv
    :param model: ранее загруженная модель
    :param img_file: путь к исходному файлу картинки
    :param out_file: путь куда записывать готовый файл
    :param blur: размывать контур сегментации
    """
    # Загрузим картинку и сменим модель цвета
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Сделаем резайз к целевому размеру
    image = img_resize_cv(image, IMG_SIZE)

    # Добавим ось
    image_np = np.expand_dims(image, axis=0)

    # Засекаем время и запускаем предикт
    time_start = time.time()
    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())
    time_end = time.time() - time_start
    print('Время предикта: {0:.1f}'.format(time_end))

    label_id_offset = 0
    # image_np_with_mask = image_np.copy()

    if 'detection_masks' in result:
        # we need to convert np.arrays to tensors
        detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
        detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

        # Reframe the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes,
            image_np.shape[1], image_np.shape[2])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        result['detection_masks_reframed'] = detection_masks_reframed.numpy()

    # Оставим только класс person == 1
    # boxes = result['detection_boxes'][0]
    classes = (result['detection_classes'][0] + label_id_offset).astype(int)
    scores = result['detection_scores'][0]
    #
    indices = np.argwhere(classes == 1)
    # boxes = np.squeeze(boxes[indices])
    classes = np.squeeze(classes[indices])
    scores = np.squeeze(scores[indices])
    #
    masks = result.get('detection_masks_reframed', None)
    masks = np.squeeze(masks[indices])

    for i in range(classes.shape[0]):
        if scores[i] > 0.9:
            image_rembg = image_np[0].copy()
            # Если задано заблюриваем контур
            if cont_blur:
                image_rembg = cut_and_blur_contour_cv(image_rembg, masks[i], cnt_thickness=10, kernel=(15, 15))
            else:
                image_rembg = apply_mask(image_rembg, masks[i])

            # Если найден не один объект, то будет несколько выходных файлов
            filename, file_extension = os.path.splitext(out_file)
            filename += '_' + str(i)
            curr_file = filename + file_extension
            print('Сохраняется {0}, scores={1:.4f}'.format(curr_file, scores[i]))
            result = Image.fromarray(image_rembg)
            result.save(curr_file)
    return


# Функция удаления фона с обработкой только opencv
def img_rem_background_cv(img_file, out_file):
    """
    Idea from https://stackoverflow.com/questions/29313667
    :param img_file: путь к исходному файлу картинки
    :param out_file: путь куда записывать готовый файл
    """

    # Загрузим картинку и сменим модель цвета
    image = cv2.imread(img_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Сделаем резайз к целевому размеру
    image = img_resize_cv(image, IMG_SIZE)

    # #
    # BLUR = 21
    # CANNY_THRESH_1 = 10
    # CANNY_THRESH_2 = 200
    # MASK_DILATE_ITER = 10
    # MASK_ERODE_ITER = 10
    # MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format
    #
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    # # -- Edge detection -------------------------------------------------------------------
    # edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    # edges = cv2.dilate(edges, None)
    # edges = cv2.erode(edges, None)
    #
    # # -- Find contours in edges, sort by area ---------------------------------------------
    # contour_info = []
    # # _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # # Previously, for a previous version of cv2, this line was:
    # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # # Thanks to notes from commenters, I've updated the code but left this note
    # for c in contours:
    #     contour_info.append((
    #         c,
    #         cv2.isContourConvex(c),
    #         cv2.contourArea(c),
    #     ))
    # contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # max_contour = contour_info[0]
    #
    # # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # # Mask is black, polygon is white
    # mask = np.zeros(edges.shape)
    # cv2.fillConvexPoly(mask, max_contour[0], (255))
    #
    # # -- Smooth mask, then blur it --------------------------------------------------------
    # mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    # mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    # mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    # mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask
    #
    # # -- Blend masked img into MASK_COLOR background --------------------------------------
    # mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    # img = image.astype('float32') / 255.0  # for easy blending
    #
    # masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    # masked = (masked * 255).astype('uint8')
##########
    # == Parameters =======================================================================
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

    # Переходим к ч/б
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    # _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = image.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    # plt.imsave('img/girl_blue.png', masked)
    # split image into channels
    c_red, c_green, c_blue = cv2.split(img)

    # merge with mask got on one of a previous steps
    img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))


    # Сохраняем изображение
    # cv2.imwrite(out_file, img_a * 255)
    cv2.imwrite(out_file, masked)
    return