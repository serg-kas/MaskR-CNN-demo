"""
Модуль с функциями
"""
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
#
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# Функция предикта object detection
def img_detection(model, img_file, out_file):

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # запускаем предикт
    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())

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

    plt.figure(figsize=(24, 32))
    plt.imshow(image_np_with_detections[0])
    # plt.show()
    plt.savefig(out_file)

    # имя и путь выходного файла сейчас не меняются
    return out_file


# Функция предикта instance segmentation
def img_segmention(model, img_file, out_file):

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # запускаем предикт
    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())

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

    plt.figure(figsize=(24, 32))
    plt.imshow(image_np_with_mask[0])
    # plt.show()
    plt.savefig(out_file)

    # имя и путь выходного файла сейчас не меняются
    return out_file


# Функция удаления фона
def img_background(model, img_file, out_file):

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # запускаем предикт
    results = model(image_np)
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())

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

    # Оставим только класс person == 1
    boxes = result['detection_boxes'][0]
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
        if scores[i] > 0.9
            image_bgrm = image_np[0].copy()
            image_bgrm = apply_mask(image_bgrm, masks[i])
            # TODO: Исправить формирование имён выходных файлов
            curr_file = out_file[:-5] + '_' + str(i) + out_file[-5:]
            print(curr_file, scores[i])
            plt.figure(figsize=(24, 32))
            plt.imshow(image_bgrm)
            plt.savefig(curr_file)

    # имя и путь выходного файла сейчас не меняются
    return out_file