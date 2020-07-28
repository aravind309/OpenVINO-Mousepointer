import cv2
import numpy as np
from math import cos, sin


def drawFaceBoundingBox(frame, faceCoords):
    cv2.rectangle(frame, (faceCoords[0], faceCoords[1]), (faceCoords[2], faceCoords[3]), (255, 255, 255), 2)


def displayHp(frame, hpOutput, faceCoords):
    yaw = hpOutput[0]
    pitch = hpOutput[1]
    roll = hpOutput[2]
    
    # To get face coordinates 
    x_max = faceCoords[2]
    x_min = faceCoords[0]
    y_max = faceCoords[3]
    y_min = faceCoords[1]
    
    # caluculating width and height for boudingbox
    bboxWidth = abs(x_max - x_min)
    bboxHeight = abs(y_max - y_min)

    x_min -= 50
    x_max += 50
    y_min -= 50
    y_max += 30
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)

    draw_axis(frame, yaw, pitch, roll, tdx=(x_min + x_max) / 2, tdy=(y_min + y_max) / 2,
              size=bboxHeight / 2)

    cv2.putText(
        frame,
        "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hpOutput[0], hpOutput[1], hpOutput[2]),
        (100, 1000),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_landmarks(frame, eyeCoords):
    cv2.rectangle(frame, (eyeCoords[0][0] - 10, eyeCoords[0][1] - 10),
                  (eyeCoords[0][2] + 10, eyeCoords[0][3] + 10), (255, 255, 255), 2)
    cv2.rectangle(frame, (eyeCoords[1][0] - 10, eyeCoords[1][1] - 10),
                  (eyeCoords[1][2] + 10, eyeCoords[1][3] + 10), (255, 255, 255), 2)


def draw_gaze(faceFrame, gazeVector, leftEye, rightEye, eyeCoords):
    x, y, w = int(gazeVector[0] * 12), int(gazeVector[1] * 12), 160
    le = cv2.line(leftEye, (x - w, y - w), (x + w, y + w), (255, 255, 255), 2)
    cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 255, 255), 2)
    re = cv2.line(rightEye, (x - w, y - w), (x + w, y + w), (255, 255, 255), 2)
    cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 255, 255), 2)
    faceFrame[eyeCoords[0][1]:eyeCoords[0][3], eyeCoords[0][0]:eyeCoords[0][2]] = le
    faceFrame[eyeCoords[1][1]:eyeCoords[1][3], eyeCoords[1][0]:eyeCoords[1][2]] = re
