import cv2
import os

PADDING = 20
def photo_to_database():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(os.getcwd() + r'\images\Vitaly_Belan\Vitaly_Belan10.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    faces_detected = "Лиц обнаружено: " + format(len(faces))
    print(faces_detected)

    for (x, y, w, h) in faces:
        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        path = os.getcwd() + r'\images\Vitaly_Belan1\Vitaly_Belan10.jpg'
        cv2.imwrite(path, gray[y1:y2, x1:x2])

    cv2.imshow("preview", image)
    cv2.waitKey(0)
    cv2.destroyWindow("preview")
    pass

if __name__ == "__main__":
    photo_to_database()