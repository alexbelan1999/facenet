from multiprocessing.dummy import Pool

from keras import backend as K

K.set_image_data_format('channels_first')
import glob
from fr_utils import *
import win32com.client as wincl
import os
from keras.models import load_model
import cv2
import time
import datetime
import inception_blocks_v2

PADDING = 20
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")
database = {}
# FRmodel = inception_blocks_v2.faceRecoModel(input_shape=(3, 96, 96))
FRmodel = load_model(os.getcwd() + '\model\FaceRecoModel.h5', None, True)


def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss


FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
# load_weights_from_FaceNet(FRmodel)
# load_weights_from_FaceNet_dump(FRmodel)
load_weights_from_FaceNet_load_from_dump(FRmodel)


def prepare_database(n):
    database = {}

    path = " "
    if n == 1:
        path = "images/Bill_Gates1/*"

    elif n == 2:
        path = "images/Elon_Musk1/*"

    elif n == 3:
        path = "images/Steve_Jobs1/*"

    elif n == 4:
        path = "images/Bill_Elon_Steve1/*"

    elif n == 5:
        path = "images/Alex_Belan/*"

    elif n == 6:
        path = "images/Vitaly_Belan/*"

    elif n == 7:
        path = "images/Alex_Vitaly/*"

    else:
        print("Ошибка варианта!")
        return database

    for file in glob.glob(path):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database


def webcam_face_recognizer(database):
    global ready_to_detect_identity

    cv2.namedWindow("web_camera")
    vc = cv2.VideoCapture(1)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while vc.isOpened():
        _, frame = vc.read()
        frame = cv2.flip(frame, 1)
        img = frame
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)

        key = cv2.waitKey(100)
        cv2.imshow("web_camera", img)
        if key == 27:
            break
    cv2.destroyWindow("web_camera")
    pass


def photo_face_recognizer(database):
    global ready_to_detect_identity
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rec = int(input("Введите 1 для фото с веб-камеры, 2 для простых фотографиях: "))
    names = None
    path = ''

    if rec == 1:
        path = "testphoto/web_camera/*"

    elif rec == 2:
        path = "testphoto/photo/*"

    else:
        print("Ошибка выбора варианта!")
    for file in glob.glob(path):
        identity = os.path.splitext(os.path.basename(file))[0]

        img = None
        if rec == 1:
            img = cv2.imread(os.getcwd() + '\\testphoto\\web_camera\\' + identity + '.jpg')

        elif rec == 2:
            img = cv2.imread(os.getcwd() + '\\testphoto\\photo\\' + identity + '.jpg')

        else:
            print('Ошибка варианта!')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        faces_detected = identity + " лиц обнаружено: " + format(len(faces))
        print(faces_detected)
        img1 = None
        if ready_to_detect_identity:
            img1 = process_frame(img, img, face_cascade)
        # cv2.imshow('photo ' + identity, img)
    cv2.waitKey(0)
    cv2.destroyWindow("photo")
    pass


def process_frame(img, frame, face_cascade):
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    identities = []
    for (x, y, w, h) in faces:
        x1 = x - PADDING
        y1 = y - PADDING
        x2 = x + w + PADDING
        y2 = y + h + PADDING

        identity = find_identity(frame, x1, y1, x2, y2)
        print(identity)
        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(frame, identity, (x1 + 5, y2 - 5), font, 1.0, (255, 255, 0), 2)
        if identity is not None:
            identities.append(identity)

    if identities != []:
        ready_to_detect_identity = False
        basename = ""
        for i in identities:
            basename += str(i) + '_'
        suffix = datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
        filename = "".join([basename, suffix])
        path = os.getcwd() + '\photo\ '.replace(' ', '') + filename + '.jpg'
        cv2.imwrite(path, img)
        ready_to_detect_identity = True
    return img


def find_identity(frame, x1, y1, x2, y2):
    height, width, channels = frame.shape
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    return who_is_it(part_image, database, FRmodel)


def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)
    identies1 = {}
    for (name, db_enc) in database.items():
        person = name[:len(name) - 2]
        if person not in identies1.keys():
            identies1[person] = 100
    mindistance = 100
    for (name, db_enc) in database.items():
        person = name[:len(name) - 2]
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' % (name, dist))
        pdist = identies1.get(person)
        if dist < pdist:
            identies1[person] = dist

    print(identies1)
    averagedis = 10
    identity = None
    for (name, dist) in identies1.items():
        if (dist < averagedis) and (dist < 0.555):
            averagedis = dist
            identity = name
    print(averagedis)
    return identity


if __name__ == "__main__":
    n = 4
    database = prepare_database(n)
    # database = {}
    # database = load.load(n)
    # webcam_face_recognizer(database)
    photo_face_recognizer(database)
