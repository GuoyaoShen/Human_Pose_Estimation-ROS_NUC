import argparse
import logging
import time

# ===update v2
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# import numpy as np
from primesense import openni2
from primesense import _openni2 as c_api
openni2.initialize("/home/quori4/OpenNI_2.3.0.55/Linux/OpenNI-Linux-x64-2.3.0.55/Tools")
dev = openni2.Device.open_any()
# depth_stream = dev.create_depth_stream()
color_stream = dev.create_color_stream()
# depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

# depth_stream.start()
color_stream.start()
# ===update v2

# import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
# ===new added
import tf_pose.utils.utils_postprocess as postprocess
# ===new added

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    # ===new added
    parser.add_argument('--tensorrt', type=str, default="False", help='for tensorrt process.')
    # ===new added
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    # logger.debug('cam read+')
    # cam = cv2.VideoCapture(args.camera)
    # ret_val, image = cam.read()
    # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        # ===update v2
        # If RGB data is needed
        cframe = color_stream.read_frame()
        cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
        R = cframe_data[:, :, 0]
        G = cframe_data[:, :, 1]
        B = cframe_data[:, :, 2]
        cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
        # cv2.imshow('color', cframe_data)

        # ret_val, image = cam.read()
        # image = cframe_data
        # ===update v2

        # ===new added
        image_copy = cframe_data.copy()
        # ===new added

        logger.debug('image process+')
        humans = e.inference(image_copy, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image_copy, humans, imgcopy=False)
        # ===new added
        # get joints coords
        list_centers = TfPoseEstimator.caclu_centers(image_copy, humans, imgcopy=False)
        print('HUMAN CENTERS LIST', list_centers)

        # get hello gesture
        postprocess.get_gesture_hello(list_centers)

        # get faces
        list_faceboxes = TfPoseEstimator.get_humanfacebox(image_copy, humans, imgcopy=False)
        print('FACES LIST', list_faceboxes)
        scalar_face = 1.0
        tuple_facesz = (128, 128)

        # show face img
        postprocess.show_face_imgs(cframe_data, list_faceboxes, tuple_facesz=tuple_facesz, scalar_face=scalar_face)
        # ===new added

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
