# Yolov5 TensorRT

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

YOLOv5 conversion and inference using TensorRT.<br>

## Convert ONNX to TensorRT

(Only supported for NVIDIA-GPUs, Tested on Linux Devices, Partial Dynamic Support)

You can convert ONNX weights to TensorRT by using the `convert.py` file. Simple run the following command: 

```
python convert.py --weights yolov5s.engine --img-size 720 1080
```

1. If using default weights, you do not need to download the ONNX model as the script will download it.

2. If you want to build the engine with custom image size, pass `--img-size custom_img_size` to `convert.py`

3. If you want to build the engine for your custom weights, simply do the following:

    - [Train Yolov5 on your custom dataset](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
    - [Export Weights PyTorch weights to ONNX](https://github.com/ultralytics/yolov5/blob/master/export.py)

    Make sure you use the `---dynamic` flag while exporting your custom weights.

    ```bash
    python export.py --weights $PATH_TO_PYTORCH_WEIGHTS --dynamic --include onnx
    ```

    Now simply use `python convert.py --weights path_to_custom_weights.onnx`, and you will have a converted TensorRT engine.
    
## Notes

- TensorRT model is not fully dynamic (for optimization reasons). You can inference on any shape of image and it'll setup engine with the first input's shape. To run inference on a different image shape, you'll have to convert a new engine.

- Building TensorRT Engine and first inference can take sometime to complete (specially if it also has to install all the dependecies for the first time).

- A new engine is built for an unseen input shape. But once built, engine file is serialized and can be used for future inference.

- Delete or rename previous serialized engine from the disk before converting a new engine (required only when image shape is different).
