import os

import cv2

cam = cv2.VideoCapture(1)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
name = input('Введите имя пользователя: ')
print("Смотрите в камеру")
count = 0
PADDING = 20

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        count += 1
        if count < 10:
            path = os.getcwd() + '\\images\\' + name + '\\' + name + '0' + str(count) + '.jpg'
        if count > 9:
            path = os.getcwd() + '\\images\\' + name + '\\' + name + str(count) + '.jpg'
        cv2.imwrite(path, gray[y1:y2, x1:x2])

    cv2.imshow('photo', img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 10:
        break

cam.release()
cv2.destroyAllWindows()
