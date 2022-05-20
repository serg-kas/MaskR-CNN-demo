"""
Модуль с функциями
"""
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import os
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

# Размер к которому приводить изображение (только функция test)
IMG_SIZE = 512

# Функция получения модели
def get_model(model_url):
    """
    :param model_name: путь к  модели для загрузки
    :return: model
    """
    if tf.test.is_gpu_available():
        print('GPU found')
    else:
        print('No GPU found')

    time_start = time.time()
    model = hub.load(model_url)
    time_end = time.time() - time_start
    print('Время загрузки модели: {0:.1f}'.format(time_end))
    return model

# Функция ресайза картинки
def img_resize(image):
    curr_w = image.shape[1]
    curr_h = image.shape[0]
    # Рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_img = IMG_SIZE / curr_w
    else:
        scale_img = IMG_SIZE / curr_h
    # Новые размеры изображения
    new_width = int(curr_w * scale_img)
    new_height = int(curr_h * scale_img)
    # делаем ресайз к целевым размерам
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image


# Функция предикта object detection
def img_detection(model, img_file, out_file):

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
def img_segmention(model, img_file, out_file):

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


# Функция удаления фона
def img_background(model, img_file, out_file):

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # Засеаем время и запускаем предикт
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

    # Функция наложения маски
    def apply_mask(image, mask):
        image[:, :, 0] = np.where(
            mask == 0,
            125,
            image[:, :, 0]
        )
        image[:, :, 1] = np.where(
            mask == 0,
            15,
            image[:, :, 1]
        )
        image[:, :, 2] = np.where(
            mask == 0,
            15,
            image[:, :, 2]
        )
        return image
    #
    # def apply_inst(image_orig, image_out, mask):
    #     image_out[:, :, 0] = np.where(
    #         mask != 0,
    #         image_orig[:, :, 0],
    #         125
    #     )
    #     image_out[:, :, 1] = np.where(
    #         mask != 0,
    #         image_orig[:, :, 1],
    #         12
    #     )
    #     image_out[:, :, 2] = np.where(
    #         mask != 0,
    #         image_orig[:, :, 2],
    #         15
    #     )
    #     return image_out

    for i in range(classes.shape[0]):
        if scores[i] > 0.9:
            image_bgrm = image_np[0].copy()
            image_bgrm = apply_mask(image_bgrm, masks[i])
            # Если найден не один объект, то будет несколько выходных файлов
            filename, file_extension = os.path.splitext(out_file)
            filename += '_' + str(i)
            curr_file = filename + file_extension
            print('Сохраняется {0}, scores={1:.4f}'.format(curr_file, scores[i]))
            result = Image.fromarray(image_bgrm)
            result.save(curr_file)
    return


# Функция удаления фона с обработкой opencv
def img_test(model, img_file, out_file):

    # image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    # image = Image.open(BytesIO(image_data))
    # (im_width, im_height) = image.size
    # image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # Загрузим картинку и сменим модель цвета
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Сделаем резайз к целевому размеру
    image = img_resize(image)

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

    # https://ru.stackoverflow.com/questions/950969
    def cut_and_blur_contour(img, mask, cnt_thickness=4, kernel=(5, 5)):
        # apply mask
        # ii = Image.fromarray(img).show()
        img = cv2.bitwise_and(img, img, mask=mask)
        tmp = img.copy()
        # ii = Image.fromarray(img).show()
        # prepare a blurred image
        blur = cv2.GaussianBlur(img, kernel, 0)


        # find contours
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours using passed [cnt_thickness] on a temporary image
        _ = cv2.drawContours(tmp, contours, 0, (0, 255, 0), cnt_thickness)

        # create contour mask
        # hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

        # apply contour mask
        tmp = cv2.bitwise_and(blur, blur, mask=mask)
        return np.where(tmp > 0, blur, img)

    for i in range(classes.shape[0]):
        if scores[i] > 0.9:
            image_bgrm = image_np[0].copy()
            image_bgrm = cut_and_blur_contour(image_bgrm, masks[i], cnt_thickness=4, kernel=(15, 15))

            # Если найден не один объект, то будет несколько выходных файлов
            filename, file_extension = os.path.splitext(out_file)
            filename += '_' + str(i)
            curr_file = filename + file_extension
            print('Сохраняется {0}, scores={1:.4f}'.format(curr_file, scores[i]))
            result = Image.fromarray(image_bgrm)
            result.save(curr_file)
    return
