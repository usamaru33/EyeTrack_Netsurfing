import cv2
import numpy as np
import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_eyes(frame, lps, rps):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if lps:
        mask = make_hole_on_mask(mask, lps)
    if rps:
        mask = make_hole_on_mask(mask, rps)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), 5)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_area = (masked_frame == [0, 0, 0]).all(axis=2)
    masked_frame[masked_area] = [255, 255, 255]
    return masked_frame

def make_hole_on_mask(mask, points):
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def get_eye_center(points):
    n = len(points)
    if n == 0:
        raise ValueError("List has no item.")
    sum_point = np.sum(points, axis=0)
    return sum_point // n

def get_eye_contours(thresh, face_mid_y, frame=None, is_right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        if is_right:
            cX += face_mid_y
        if frame is not None:
            cv2.circle(frame, (cX, cY), 2, (255, 0, 0), 2)
        return (cX, cY)
    except ValueError:
        pass
    except ZeroDivisionError:
        pass
    return None