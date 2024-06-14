import cv2
import numpy as np
from pynput.mouse import Controller
from face_detect import get_one_face
from eye_detect import extract_eyes, get_eye_center, get_eye_contours
import dlib
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

# dlibの顔特徴点検出器を初期化
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# マウスコントローラーの初期化
mouse = Controller()

def scroll_based_on_eye_position(eye_center, frame_center):
    """
    黒目の中心位置に基づいてスクロールを実行する関数
    """
    diff_x = eye_center[0] - frame_center[0]
    if diff_x > 15:
        mouse.scroll(0, -1)  # 下方向にスクロール
    elif diff_x < -15:
        mouse.scroll(0, 1)  # 上方向にスクロール

def main():
    """
    メイン関数：カメラから映像を取得し、黒目の位置を検出してスクロール操作を行う
    """
    cap = cv2.VideoCapture(0)  # カメラのキャプチャを開始
    while True:
        ret, frame = cap.read()  # フレームをキャプチャ
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # フレームをグレースケールに変換
        face = get_one_face(gray_frame)  # 顔を検出
        if face is not None:
            # 顔のランドマーク（特徴点）を取得
            landmarks = predictor(gray_frame, face)
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            
            # 右目の部分を抽出
            masked_frame = extract_eyes(frame, [], right_eye_points)
            eyes_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)  # 再度グレースケールに変換
            
            # 閾値処理
            threshold = 50  # ここを調整して最適な値を見つける
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            thresh = cv2.medianBlur(thresh, 3)
            thresh_inv = cv2.bitwise_not(thresh)
            
            # 顔の中央のy座標を計算
            face_mid_y = (face.left() + face.right()) // 2
            
            # 右目の黒目の中心を検出
            right_eye_center = get_eye_contours(thresh_inv, face_mid_y, frame, is_right=True)
            
            if right_eye_center:
                logging.info(f"Right eye center detected at: {right_eye_center}")
                # 右目の黒目の中心を緑色で表示
                cv2.circle(frame, right_eye_center, 2, (0, 255, 0), 2)
                
                # フレームの中心を計算
                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                # スクロール操作を実行
                scroll_based_on_eye_position(right_eye_center, frame_center)
        
        cv2.imshow("Eye Tracking", frame)  # フレームを表示
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'キーが押されたらループを終了
            break
    cap.release()  # カメラのキャプチャを解放
    cv2.destroyAllWindows()  # すべてのOpenCVウィンドウを閉じる

if __name__ == "__main__":
    main()