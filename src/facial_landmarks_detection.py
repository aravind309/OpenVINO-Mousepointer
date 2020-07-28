import numpy as np
from openvinoHelper import Inference


class Facial_Landmarks_Detection(Inference):
    def __init__(self):
        super().__init__()

    def preprocess_output(self, outputs, *args, **kwargs):
        result = outputs[self.outputBlob][0]
        leftEye_x, leftEye_y, rightEye_x, rightEye_y = [result[i][0][0] for i in range(4)]

        return leftEye_x, leftEye_y, rightEye_x, rightEye_y

    def predict(self, image):
        coords = self.prediction_helper(image, self.preprocess_output)
        h, w = image.shape[0], image.shape[1]

        coords = (coords * np.array([w, h, w, h])).astype(np.int32)

        l_eye_xmin, l_eye_ymin, r_eye_xmin, r_eye_ymin = np.array(coords) - 10
        l_eye_xmax, l_eye_ymax, r_eye_xmax, r_eye_ymax = np.array(coords) + 10

        left_e = image[l_eye_ymin:l_eye_ymax, l_eye_xmin:l_eye_xmax]
        right_e = image[r_eye_ymin:r_eye_ymax, r_eye_xmin:r_eye_xmax]

        eyeCoords = [[l_eye_xmin, l_eye_ymin, l_eye_xmax, l_eye_ymax], [r_eye_xmin, r_eye_ymin, r_eye_xmax, r_eye_ymax]]

        return left_e, right_e, eyeCoords
