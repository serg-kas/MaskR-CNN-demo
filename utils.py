"""
Модуль с функциями
"""
# import cv2
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from six.moves.urllib.request import urlopen
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
                               (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
                               (11, 13), (13, 15), (12, 14), (14, 16)]

# Размер к которому приводить изображение
# IMG_SIZE = 1024
#


# Функция предикта одной картинки
def img_predict(operation_mode, model, img_file, out_file):

    image_data = tf.io.gfile.GFile(img_file, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((1, im_height, im_width, 3)).astype(np.uint8)

    # running inference
    results = model(image_np)

    # different object detection models have additional results
    # all of them are explained in the documentation
    result = {key: value.numpy() for key, value in results.items()}
    # print(result.keys())

    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in result:
        keypoints = result['detection_keypoints'][0]
        keypoint_scores = result['detection_keypoint_scores'][0]

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections[0],
        result['detection_boxes'][0],
        (result['detection_classes'][0] + label_id_offset).astype(int),
        result['detection_scores'][0],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

    plt.figure(figsize=(24, 32))
    plt.imshow(image_np_with_detections[0])
    # plt.show()
    plt.savefig(out_file)

    # имя и путь выходного файла сейчас не меняются
    return out_file
