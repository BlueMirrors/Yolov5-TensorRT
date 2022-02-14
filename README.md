# Yolov5 TensorRT

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tXLk2KFZkXQ7SpTBbmQ_Y43Eo1_34Rsf?usp=sharing)

YOLOv5 conversion and inference using TensorRT (FP16), with no complicated installations setup and zero precession loss!

- [FPS Info](#fps-and-accuracy-info)

- [Google-Colab](https://colab.research.google.com/drive/1tXLk2KFZkXQ7SpTBbmQ_Y43Eo1_34Rsf?usp=sharing)

## Inference with TensorRT

Tested with Linux based systems (Colab T4/P4/K80, Jetson-Nano (with jetpack installed), Ubuntu-GTX 1650)

First clone this repo and install requirements

```bash
$ git clone https://github.com/BlueMirrors/Yolov5-TensorRT.git
$ cd Yolov5-TensorRT
$ pip install -r requirements.txt
```

Now run inference on video or image file (with pretrained weights).

```bash
python detect.py --input $PATH_TO_INPUT_FILE --output $OUTPUT_FILE_NAME
```

<br>

You can also pass ```--weights``` to use your own custom onnx weight file (it'll generate tensorrt engine file internally) or tensorrt engine file (generated from convert.py). You can also pass ```--classes``` for your custom trained weights and/or to filter classes for COCO.

For pretrained default weights (```--weights yolov5s```), scripts will download + internally generate new engine file for unseen input shape, but if you are using a custom weight then remeber to rename or remove engine file if you want to generate engines for different shapes. 

## Convert ONNX to TensorRT

(Only supported for NVIDIA-GPUs, Tested on Linux Devices, Partial Dynamic Support)

You can convert ONNX weights to TensorRT by using the `convert.py` file. Simple run the following command: 

```
python convert.py --weights yolov5s.engine --img-size 720 1080
```

1. By default the onnx model is converted to TensorRT engine with FP16 precision. To convert to TensorRT engine with FP32 precision use ```--fp32``` when running the above command.

2. If using default weights, you do not need to download the ONNX model as the script will download it.

3. If you want to build the engine with custom image size, pass `--img-size custom_img_size` to `convert.py`

4. If you want to build the engine for your custom weights, simply do the following:

    - [Train Yolov5 on your custom dataset](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
    - [Export Weights PyTorch weights to ONNX](https://github.com/ultralytics/yolov5/blob/master/export.py)

    Make sure you use the `---dynamic` flag while exporting your custom weights.

    ```bash
    python export.py --weights $PATH_TO_PYTORCH_WEIGHTS --dynamic --include onnx
    ```

    Now simply use `python convert.py --weights path_to_custom_weights.onnx`, and you will have a converted TensorRT engine. Also add ```--nc``` (number of classes) if your custom model has different number of classes than COCO(i.e. 80 classes). 
    
## FPS and Accuracy Info
***In our tests, TensorRT had identical outputs as original pytorch weights.***

Based on 5000 inference iterations after 100 iterations of warmups. Includes Image Preprocessing (letterboxing etc.), Model Inference and Output Postprocessing (NMS, Scale-Coords, etc.) time only.  

| Hardware    | FPS     |
| ---------- | ------- |
| T4   | 157-165 |
| GTX 1650 | 138-145|
| P4   | 82-86 |
| K80 | 49-55 | 
    
## Notes
- We support Letterboxing (significantly better accuracy!!)
- TensorRT model is not fully dynamic (for optimization reasons). You can inference on any shape of image and it'll setup engine with the first input's shape. To run inference on a different image shape, you'll have to convert a new engine.

- Building TensorRT Engine and first inference can take sometime to complete (specially if it also has to install all the dependecies for the first time).

- A new engine is built for an unseen input shape. But once built, engine file is serialized and can be used for future inference.

- Delete or rename previous serialized engine from the disk before converting a new engine (required only when image shape is different).
- Batch support will be added next week (after 15th August)

## References
- [Yolov5](https://github.com/ultralytics/yolov5)
- [CVU](https://github.com/BlueMirrors/cvu)
