import os
import time
import argparse
import numpy as np
from cvu.utils.google_utils import gdrive_download
from cvu.detector.yolov5 import Yolov5 as Yolov5Trt


def convert_onnx_to_trt(onnx_weights, image_shape, nc):
    if isinstance(image_shape, int) or isinstance(image_shape, list):
        image_shape = tuple(image_shape)
    if len(image_shape) == 1:
        image_shape = (image_shape[0], image_shape[0], 3)
    elif len(image_shape) == 2:
        image_shape += (3, )

    start = time.time()
    convert = Yolov5Trt(classes=['0'] * nc,
                        weight=onnx_weights,
                        backend="tensorrt")
    print(image_shape)
    convert(np.random.randint(0, 255, image_shape).astype("float"))
    print("\n\nTotal Time Taken: ", round(time.time() - start, 2), "seconds.")
    print("Engine File Saved at: ", onnx_weights.replace(".onnx", ".engine"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='yolov5s.onnx',
                        help='weights path')

    parser.add_argument('--img-size',
                        nargs='+',
                        type=int,
                        default=[720, 1280],
                        help='image (height, width)')

    parser.add_argument('--nc', type=int, default=80, help='number of classes')

    opt = parser.parse_args()

    print(opt.weights)
    if not os.path.exists(opt.weights):
        if opt.weights == 'yolov5s.onnx':
            print(f"Warnning: {opt.weights} not found,",
                  "downloading pretrained weights...")
            gdrive_download("1piC3ZGuc4D8MMJQQRK3dgaCa66-4Ucxi",
                            "yolov5s.onnx")
        else:
            raise FileNotFoundError(
                (f"ONNX weight file not found at {opt.weights}." +
                 " Please check again."))

    convert_onnx_to_trt(opt.weights, image_shape=opt.img_size, nc=opt.nc)
