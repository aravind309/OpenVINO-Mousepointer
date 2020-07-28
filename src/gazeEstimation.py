import cv2
import numpy as np
import math
from openvinoHelper import Inference


# Reference: https://github.com/mdfazal/computer-pointer-controller-1/blob/master/gazeEstimation.py
class gazeEstimation(Inference):
    def __init__(self):
        super().__init__()

    def preprocess_input(self, leftEye_image, rightEye_image):
        re_resized = cv2.resize(rightEye_image, (self.get_gaze_input_shape()[3], self.get_gaze_input_shape()[2]))
        re_processed = np.transpose(np.expand_dims(re_resized, axis=0), (0, 3, 1, 2))

        le_resized = cv2.resize(leftEye_image, (self.get_gaze_input_shape()[3], self.get_gaze_input_shape()[2]))
        le_processed = np.transpose(np.expand_dims(le_resized, axis=0), (0, 3, 1, 2))

        return re_processed, le_processed

    def preprocess_output(self, outputs, head_pose_angle):
        gaze_vec = outputs[self.outputBlob[0]][0]
        angle_r_fc = head_pose_angle[2]
        cosine = math.cos(angle_r_fc * math.pi / 180.0)
        sine = math.sin(angle_r_fc * math.pi / 180.0)

        x_val = gaze_vec[0] * cosine + gaze_vec[1] * sine
        y_val = -gaze_vec[0] * sine + gaze_vec[1] * cosine

        return (x_val, y_val), gaze_vec

    def predict(self, leftEye, rightEye, head_pose_angle):
        le_img_processed, re_img_processed = self.preprocess_input(leftEye.copy(), rightEye.copy())

        result = self.execNetwork.infer({'head_pose_angles': head_pose_angle, 'leftEye_image': le_img_processed,
                                          'rightEye_image': re_img_processed})

        mouse_coords, gaze_vec = self.preprocess_output(result, head_pose_angle)

        return mouse_coords, gaze_vec
