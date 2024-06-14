import numpy as np
import dlib
import cv2
from cv2 import Mat
#顔の特徴点検出モデルの読み込み
face_detector = dlib.get_frontal_face_detector()
# Dlibの顔ランドマーク検出器モデルの読み込み
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#カメラの設定
cap = cv2.VideoCapture(0)  # 0はデフォルトカメラ
ret, frame = cap.read()

#マスク色
bgr_black = (0, 0, 0)  # BGR黒色
bgr_white = (255, 255, 255)  # BGR白色

# 検出点の色定義
bgr_blue = (255, 0, 0)

# 点の座標を表すためのPointクラス
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @classmethod
    def from_contour(cls, cnt):
        # cnt が空の場合は None を返す
        if len(cnt) == 0:
            return None
        
        # 輪郭の最小外接円の中心座標を取得
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = cls(int(x), int(y))
        return center
    
# 前回の目の座標を保持する (左目、右目)
get_contouring.prev_points = [None, None]

# 評価関数 (前回の座標からの距離が小さいほど高スコア)
def eval_contour(cnt, prev):
    if prev is None:
        return 0
    return 1.0 / (cv2.pointPolygonTest(cnt, (prev.x, prev.y), True) + 1)

def get_one_face(gray_image: Mat) -> Mat | None:
    faces = face_detector(gray_image)
    if len(faces) == 0:
        return None
    face = faces[0]

    return face

gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
face = get_one_face(gray_frame)

display = frame.copy()

if face is not None:
    landmarks = predictor(gray_frame, face) # 特徴点を表す座標のlist(len=68)

def extract_eyes(frame: Mat, lps: list[Point], rps: list[Point]) -> Mat:
    # make mask with eyes white and other black
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask = make_hole_on_mask(mask, lps)
    mask = make_hole_on_mask(mask, rps)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), 5)

    # attach mask on frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    masked_area = (masked_frame == bgr_black).all(axis=2)
    masked_frame[masked_area] = bgr_white
    return masked_frame # 目以外が白塗りされた画像


# 目の輪郭を検知して円を描画する関数
def get_contouring(
    thresh, face_mid_y, frame=None, is_right: bool = False
) -> Point | None:
    index = int(is_right)
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    try:
        cnt = max(
            cnts,
            key=lambda cnt: eval_contour(
                cnt, get_contouring.prev_points[index]
            ),
        )
        c = Point.from_contour(cnt)
        if c is None:
            raise ValueError("point is Null.")
        if is_right:
            c.x += face_mid_y
        get_contouring.prev_points[index] = c
        if frame is not None:
            cv2.circle(frame, (c.x, c.y), 2, bgr_blue, 2)
        return Point(c.x, c.y)
    except ValueError:
        pass
    except ZeroDivisionError:
        pass