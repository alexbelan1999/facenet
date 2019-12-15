import glob
import os

import cv2

PADDING = 20
identity = ''


def photo_to_database():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_id = int(input('Введите 1 - Bill_Gates, 2 - Elon_Musk, 3 - Steve_Jobs , 4 - Bill_Elon_Musk: '))

    path = ''
    if face_id == 1:
        path = "images/Bill_Gates/*"

    elif face_id == 2:
        path = "images/Elon_Musk/*"

    elif face_id == 3:
        path = "images/Steve_Jobs/*"

    elif face_id == 4:
        path = "images/Bill_Elon_Steve/*"

    else:
        print('Ошибка варианта!')

    for file in glob.glob(path):
        global identity
        identity = os.path.splitext(os.path.basename(file))[0]
        img = None
        if face_id == 1:
            img = cv2.imread(os.getcwd() + '\\images\\Bill_Gates\\' + identity + '.jpg')

        elif face_id == 2:
            img = cv2.imread(os.getcwd() + '\\images\\Elon_Musk\\' + identity + '.jpg')

        elif face_id == 3:
            img = cv2.imread(os.getcwd() + '\\images\\Steve_Jobs\\' + identity + '.jpg')

        elif face_id == 4:
            img = cv2.imread(os.getcwd() + '\\images\\Bill_Elon_Steve\\' + identity + '.jpg')

        else:
            print('Ошибка варианта!')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        faces_detected = identity + " лиц обнаружено: " + format(len(faces))
        print(faces_detected)

        for (x, y, w, h) in faces:
            x1 = x - PADDING
            y1 = y - PADDING
            x2 = x + w + PADDING
            y2 = y + h + PADDING
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            path1 = ' '

            if face_id == 1:
                path1 = os.getcwd() + '\\images\\Bill_Gates1\\' + identity + '.jpg'

            elif face_id == 2:
                path1 = os.getcwd() + '\\images\\Elon_Musk1\\' + identity + '.jpg'

            elif face_id == 3:
                path1 = os.getcwd() + '\\images\\Steve_Jobs1\\' + identity + '.jpg'

            elif face_id == 4:
                path1 = os.getcwd() + '\\images\\Bill_Elon_Steve1\\' + identity + '.jpg'

            else:
                print('Ошибка варианта!')

            cv2.imwrite(path1, gray[y1:y2, x1:x2])
            cv2.imshow('photo' + identity, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


if __name__ == "__main__":
    photo_to_database()
