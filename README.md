# Mouse Pointer Controller
The application with help Intel OpenVINO uses gaze detection to move the mouse pointer position.This application has the capability to accept video streams and live feeds. Please find the sample output in output folder.


## Project Set Up and Installation
This is tested on Ubuntu 16 and Ubuntu 18 with FP32 and FP16 precesions on combination of CPU,iGPU and MYRIAD . Hardware device used is upsquared  core board.
Python version used is 3.7

 This current repo is tested on both Ubuntu 18 and Ubuntu 16 with Intel® Distribution of OpenVINO™ toolkit 2020.2. For installation refer to[Open Vino installation](https://docs.openvinotoolkit.org/2020.2/_docs_install_guides_installing_openvino_linux.html). 

#### Model optimizer

The Model Optimizer of OpenVINO helps us to convert models(.pb files, .onnx files etc) in multiple different frameworks to IR files which will be used Inference Engine of OpenVINO. The main adavntage of using Model optimizer is it will optimize the models by shriking model size and accelerates the speed.
The model optimizer supports at present INT8, FP32, FP16 outputs. The is always a compramise between model size and speed. If model size is large(higher precision) speed willbe lower but the accuracy will be higher.
OpenVINO™ toolkit and its dependencies must be installed to run the application. OpenVINO 2020.2.130 is used on this project.  
 Installation instructions may be found at:
 * https://software.intel.com/en-us/articles/OpenVINO-Install-Linux 
 * https://github.com/udacity/nd131-openvino-fundamentals-project-starter/blob/master/linux-setup.md  

Download the below pretrained models as described in the setup instructions.

`source /opt/intel/openvino/bin/setupvars.sh`

`cd $PROJ_DIR/`
 
 PROJ_DIR is a directory where we cloned the current project.

-   [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_faceDetection_adas_binary_0001_description_faceDetection_adas_binary_0001.html)
-   [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
-   [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
-   [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gazeEstimation_adas_0002_description_gazeEstimation_adas_0002.html)



*Face Detection*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```

*Head Pose Estimation*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
```

*Gaze Estimation Model*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
```

*Facial Landmarks Detection*

```
$ python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
```

*Project structure*

```
|--bin
    |--demo.mp4
|--intel
|--output
    |--MousePointer.mp4
|--src
    |--faceDetection.py
    |--facial_landmarks_detection.py
    |--gazeEstimation.py
    |--headPose.py
    |--input_feeder.py
    |--main.py
    |--mouse_controller.py
    |--openvinoHelper.py
    |--predictionVisualization.py
|--README.md
|--requirements.txt
```


All the code files are present in src.

src folder contains all the source files:-

1. faceDetection.py: This file is to prpreprocess the video frame and perform inference to detect faces.

2. facial_landmarks_detection.py: The output of faceDetection.py is fed as input.This file helps us in detecting the eye landmarks.

3. headPose.py The output of faceDetection.py is fed as input.This file helps us in detecting yaw-roll-pitch angles. 

4. gazeEstimation.py: It will take the left eye, rigt eye, head pose angles as inputs, perform inference and predicts the gaze vector.

5. openvinoHelper.py: This file is used by all the above files as base class and has openvino functions. Its purpose is to avoid code redundancy.

6. input_feeder.py: This is for assitance to users input param and return frame one after another. 

7. mouse_controller.py:Contains MouseController class which take x, y coordinates value, speed, precisions and moves the mouse accordingly.

8. predictionVisualization.py: Helps for visualization like writing text and other things on frame.

9. main.py: Driver file


## Demo
Use the following command to run the application

```
python3 src/main.py -i 'bin/demo.mp4' -fd 'models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml' -ld 'models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'  -ge 'models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml' -hp 'models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml' 
```

Please experiment with changing the precisions available in intel directory.

## Documentation

command line parameters
```
$ python3 main.py --help
usage: main.py [-h] -fd faceDetectionModel -ld faceLandmarkModel -ge
               gazeEstimationModel -hp headPoseModel -i INPUT
              [-prob PROB_THRESHOLD] [-d DEVICE]


optional arguments:
  -h, --help            show this help message and exit
  -fd faceDetectionModel, --faceDetectionModel faceDetectionModel
                        Path to .xml file of Face Detection model.
  -ld faceLandmarkModel, --faceLandmarkModel faceLandmarkModel
                        Path to .xml file of Facial Landmark Detection model.
  -ge gazeEstimationModel, --gazeEstimationModel gazeEstimationModel
                        Path to .xml file of Gaze Estimation model.
  -hp headPoseModel, --headPoseModel headPoseModel
                        Path to .xml file of Head Pose Estimation model.
  -i INPUT, --input INPUT
                        Path to video file or enter cam for webcam
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        path of extensions if any layers is incompatible with
                        hardware
  -prob PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for model to identify the face .
  -d DEVICE, --device DEVICE
                        Specify the target device to run on: CPU, GPU, FPGA or
                        MYRIAD is acceptable. Sample will look for a suitable
                        plugin for device (CPU by default)
  -v VISUALIZATION, --visualization VISUALIZATION
                        Set to True to visualization all different model
                        outputs Made it as default
```
 The below benchmarks are taken on core based device(1.97GHz and 8 core). These are collected

# Benchmarks
* Gaze Estimation Model
  | Precision level | FPS     | Latency(ms) | Total Execution Time (sec) |
  |-----------------|---------|-------------|--------------------------- |
  | INT 8           |  320.21 |  2.83       |  18.93                 |
  | FP16            |  180.59 |  4.56       |  19.67                 |
  | FP32            | 165.89 |  5.03       |  20.42                  |
* Face Detection Model
  | Precision level | FPS    | Latency(ms) | Total Execution Time (sec) |
  |-----------------|--------|-------------|--------------------------- |
  | INT 8           |  14.68 |  55.47      |  18.06                 |
  | FP16            |  12.29 |  61.56      |  19.12                  |
  | FP32            |  12.09 |  63.20      | 19.94                  |
* Head Pose Estimation Model
  | Precision level | FPS     | Latency(ms) | Total Execution Time (sec) | 
  |-----------------|---------|-------------|--------------------------- |
  | INT 8           |  378.2  |  67.23      |  18.54                  |
  | FP16            |  301.45 |  71.29      |  20.34                  |
  | FP32            |  289.78 |  74.49      |  20.91                  |
* Landmarks Detection Model
  | Precision level | FPS     | Latency(ms) | Total Execution Time (sec) |
  |-----------------|---------|-------------|--------------------------- |
  | INT 8           | 1500.75 |  1.42      |  18.12                   |
  | FP16            | 1477.72 |  1.56       |  19.58                  |
  | FP32            | 1463.25 |  1.57       | 19.91



## Results
From the above results, the best model precision combination is that of Face detection 32 bits precision with other models in 16 bits. 
This reduce the model size and load time, although models with lower precision gives low accuracy but better inference time.

## Stand Out Suggestions

While there is well known relation between Precesion and accuracy I preferred combination of FP16 and FP32 models for a better output.
To achieve overall best inference rate FP16 is suggested and it this selection at not at the compramise of accuracy. 


### Edge Cases
Below are the edge cases that breaks inference flow.

1. Incase if the scrolling goes out of range the inference pipeline will break.
2. Lighting conditions may effect the gaze condition.
3. In case of multiple people in frame it is using the first detected person as track pointer.



