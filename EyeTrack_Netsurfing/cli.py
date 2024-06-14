import cv2
threshold = 30
_, thresh = cv2.threshold(
	eyes_frame_gray, threshold, 255, cv2.THRESH_BINARY
)
thresh_erode = cv2.erode(thresh, None, iterations=2)
thresh_dilate = cv2.dilate(thresh_erode, None, iterations=4)
thresh_blur = cv2.medianBlur(thresh_dilate, 3)
thresh_inv = cv2.bitwise_not(thresh_blur)