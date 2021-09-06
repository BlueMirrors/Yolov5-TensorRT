"""Run Inference using TensorRT.

Args
    --weights: onnx weights path or trt engine path; default yolov5s will download pretrained weights.
    --inputs: path to input video or image file. default people.mp4 will download demo video.
    --output: path to output video or image file. default out.mp4 (out.jpg if image file given in input )
"""
import argparse
import os
import time
import numpy as np
import cv2
from cvu.detector.yolov5 import Yolov5 as Yolov5Trt
from vidsz.opencv import Reader, Writer
from cvu.utils.google_utils import gdrive_download


def detect_video(weight,
                 input_video,
                 output_video=None,
                 classes="coco",
                 auto_install=True):

    # load model
    model = Yolov5Trt(classes=classes,
                      backend="tensorrt",
                      weight=weight,
                      auto_install=auto_install)

    # setup video reader and writer
    reader = Reader(input_video)
    writer = Writer(reader,
                    name=output_video) if output_video is not None else None

    # warmup
    warmup = np.random.randint(0, 255, reader.read().shape).astype("float")
    for i in range(100):
        model(warmup)

    inference_time = 0
    for frame in reader:
        # inference
        start = time.time()
        preds = model(frame)
        inference_time += time.time() - start

        # draw on frame
        if writer is not None:
            preds.draw(frame)
            writer.write(frame)

    print("\nModel Inference FPS: ",
          round(reader.frame_count / inference_time, 3))
    print("Output Video Saved at: ", writer.name)
    writer.release()
    reader.release()


def detect_image(weight,
                 image_path,
                 output_image,
                 classes="coco",
                 auto_install=True):
    # load model
    model = Yolov5Trt(classes=classes,
                      backend="tensorrt",
                      weight=weight,
                      auto_install=auto_install)

    # read image
    image = cv2.imread(image_path)

    # inference
    preds = model(image)
    print(preds)

    # draw image
    preds.draw(image)

    # write image
    print(output_image)
    cv2.imwrite(output_image, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        default='yolov5s',
                        help='onnx weights path or trt engine path')

    parser.add_argument('--input',
                        type=str,
                        default='people.mp4',
                        help='path to input video or image file')

    parser.add_argument('--output',
                        type=str,
                        default='out.mp4',
                        help='name of output video or image file')

    parser.add_argument('--classes',
                        nargs='+',
                        default=None,
                        type=str,
                        help=(('custom classes or filter coco classes ' +
                               'classes: --class car bus person')))

    parser.add_argument('--no-auto-install',
                        action='store_true',
                        help="Turn off auto install feature")

    opt = parser.parse_args()

    if opt.classes is None:
        opt.classes = 'coco'

    # image file
    input_ext = os.path.splitext(opt.input)[-1]
    output_ext = os.path.splitext(opt.output)[-1]

    if input_ext in (".jpg", ".jpeg", ".png"):
        if output_ext not in ((".jpg", ".jpeg", ".png")):
            opt.output = opt.output.replace(output_ext, input_ext)
        detect_image(opt.weights, opt.input, opt.output, opt.classes,
                     not opt.no_auto_install)

    # video file
    else:
        if not os.path.exists(opt.input) and opt.input == 'people.mp4':
            gdrive_download("1rioaBCzP9S31DYVh-tHplQ3cgvgoBpNJ", "people.mp4")

        detect_video(opt.weights, opt.input, opt.output, opt.classes,
                     not opt.no_auto_install)
