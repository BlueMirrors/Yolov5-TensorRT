import os
import time
import argparse
import numpy as np
from cvu.utils.google_utils import gdrive_download
from cvu.detector.yolov5 import Yolov5 as Yolov5Trt


def convert_onnx_to_trt(onnx_weights, image_shape, nc, fp16, auto_install=True):
    # sanity check image shape
    if isinstance(image_shape, int) or isinstance(image_shape, list):
        image_shape = tuple(image_shape)
    if len(image_shape) == 1:
        image_shape = (image_shape[0], image_shape[0], 3)
    elif len(image_shape) == 2:
        image_shape += (3, )

    start = time.time()
    convert = Yolov5Trt(classes=list(map(str, range(nc))),
                        weight=onnx_weights,
                        backend="tensorrt",
                        auto_install=auto_install,
                        fp16=fp16)
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
    parser.add_argument('--no-auto-install',
                        action='store_true',
                        help="Turn off auto install feature")
    parser.add_argument('--fp32',
                        action='store_true',
                        help="Create yolov5 engine with FP32 precision")

    opt = parser.parse_args()

    print(opt.weights)

    # check if engine already exists
    if os.path.exists(opt.weights.replace("onnx", "engine")):
        print("Engine Already Exists. Please rename it",
              "or remove it to build a new engine from scratch.")
        exit()

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

    convert_onnx_to_trt(opt.weights,
                        image_shape=opt.img_size,
                        nc=opt.nc,
                        fp16=not opt.fp32,
                        auto_install=not opt.no_auto_install)