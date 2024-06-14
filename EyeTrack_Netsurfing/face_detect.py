import dlib

face_detector = dlib.get_frontal_face_detector()

def get_one_face(gray_image):
    faces = face_detector(gray_image)
    if len(faces) == 0:
        return None
    return faces[0]