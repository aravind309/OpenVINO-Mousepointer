import os
import cv2
import time
import numpy as np
from argparse import ArgumentParser
from faceDetection import FaceDetection
from input_feeder import InputFeeder
from gazeEstimation import gazeEstimation
from facial_landmarks_detection import Facial_Landmarks_Detection
from headPose import headPose
from predictionVisualization import drawFaceBoundingBox, displayHp, draw_landmarks, draw_gaze
from mouse_controller import MouseController


def build_argparser():
    parser = ArgumentParser()

    parser.add_argument("-fd", "--faceDetectionModel", required=True, type=str,
                        help=" Path to .xml file of Face Detection model.")
    parser.add_argument("-ld", "--faceLandmarkModel", required=True, type=str,
                        help=" Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headPoseModel", required=True, type=str,
                        help=" Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-ge", "--gazeEstimationModel", required=True, type=str,
                        help=" Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help=" Path to video file or enter cam for webcam")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to identify the face .")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to run on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "(CPU by default)")
    parser.add_argument("-v", "--visualization", required=False, type=bool,
                        default=True,
                        help="Set to False to disable visualization for all different model outputs")

    return parser


def main():
    args = build_argparser().parse_args()

    frameNum = 0
    inferenceTime = 0
    counter = 0

    # Initialize the Inference Engine
    fd = FaceDetection()
    ld = Facial_Landmarks_Detection()
    ge = gazeEstimation()
    hp = headPose()
    modelStart = time.time()
    # Load Models
    fd.loadModel(args.faceDetectionModel, args.device)
    ld.loadModel(args.faceLandmarkModel, args.device)
    ge.loadModel(args.gazeEstimationModel, args.device)
    hp.loadModel(args.headPoseModel, args.device)
    print("Model Load timing:",(time.time()-modelStart)*1000,"ms")

    # Get the input feeder
    if args.input == "cam":
        feed = InputFeeder("cam")
    else:
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
        feed = InputFeeder("video", args.input)
    feed.load_data()
    frameCount = 0
    # Mouse Controller precision and speed
    mc = MouseController('medium', 'fast')

    for frame in feed.next_batch():
        frameCount += 1
        
        if frame is not None:
            key = cv2.waitKey(60)

            inferenceStart = time.time()

            # make predictions
            detected_face, faceCoords = fd.predict(frame.copy(), args.prob_threshold)
            hpOutput = hp.predict(detected_face.copy())
            leftEye, rightEye, eyeCoords = ld.predict(detected_face.copy())
            new_mouse_coord, gazeVector = ge.predict(leftEye, rightEye, hpOutput)

            inferenceTime = time.time()- inferenceStart
            counter = counter + 1

            # Visualization
            preview = args.visualization
            if preview:
                preview_frame = frame.copy()
                faceFrame = detected_face.copy()

                drawFaceBoundingBox(preview_frame, faceCoords)
                displayHp(preview_frame, hpOutput, faceCoords)
                draw_landmarks(faceFrame, eyeCoords)
                draw_gaze(faceFrame, gazeVector, leftEye.copy(), rightEye.copy(), eyeCoords)
            if preview:
                img = np.hstack((cv2.resize(preview_frame, (500, 500)), cv2.resize(faceFrame, (500, 500))))
            else:
                img = cv2.resize(frame, (500, 500))
            cv2.imshow('Visualization', img)

            # set speed
            if frameCount % 5 == 0:
                mc.move(new_mouse_coord[0], new_mouse_coord[1])


            print("Frame Number:",frameNum)
            print("Inference Time:",inferenceTime*1000)

            frameNum += 1

            if key == 27:
                break
    feed.close()
if __name__ == '__main__':
    main()
