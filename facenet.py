from keras import backend as K
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import glob
import numpy as np
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl
import os
from keras.models import load_model
import time
import load

PADDING = 50
ready_to_detect_identity = True
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice")
database = {}
print("Start", (time.ctime(time.time())))
#FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("FRmodel", (time.ctime(time.time())))
FRmodel = load_model(os.getcwd() + '\model\FaceRecoModel.h5')

def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss
print("FRmodel.compile", (time.ctime(time.time())))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
print("load_weights_from_FaceNet", (time.ctime(time.time())))
load_weights_from_FaceNet(FRmodel)
print("end load_weights_from_FaceNet", (time.ctime(time.time())))
def prepare_database(n):
    database = {}
    arr = [arr for arr in range(1, 5)]
    if n not in arr:
        return database
    str = " "
    if n == 1:
        str = "images/Bill_Gates/*"

    if n == 2:
        str = "images/Elon_Musk/*"

    if n == 3:
        str = "images/Steve_Jobs/*"

    if n == 4:
        str = "images/Bill_Elon_Steve/*"

    for file in glob.glob(str):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_path_to_encoding(file, FRmodel)

    return database

def webcam_face_recognizer(database):
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while vc.isOpened():
        _, frame = vc.read()
        frame = cv2.flip(frame, 1)
        img = frame
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)   
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)
        if key == 27:
            break
    cv2.destroyWindow("preview")
    pass

def photo_face_recognizer(database):
    global ready_to_detect_identity
    print("start")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    frame = cv2.imread(os.getcwd() + r'\testphoto\test1.jpg')
    img = frame
    if ready_to_detect_identity:
        img = process_frame(img, frame, face_cascade)

    cv2.imshow("preview", img)
    cv2.waitKey(0)
    cv2.destroyWindow("preview")
    print("end")
    pass

def process_frame(img, frame, face_cascade):
    print("process_frame")
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        identity = find_identity(frame, x1, y1, x2, y2)
        print(identity)
        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(frame, identity, (x1, y2), font, 1.0, (255, 255, 255), 1)
        if identity is not None:
            cv2.imwrite(os.getcwd() + '\photo\ '.replace(' ','') + identity + '.png', img)
            identities.append(identity)

    if identities != []:

        ready_to_detect_identity = False
        pool = Pool(processes=1)
        pool.apply_async(welcome_users, [identities])
    return img

def find_identity(frame, x1, y1, x2, y2):
    print("find_identity")
    height, width, channels = frame.shape
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    
    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    print("who_is_it")
    encoding = img_to_encoding(image, model)
    identies = {}
    for (name, db_enc) in database.items():
        str = name[:len(name) - 2]
        if str not in identies.keys():
            identies[str]= 0

    min_dist = 100
    sumdist = 0
    identity = None
    for (name, db_enc) in database.items():
        str = name[:len(name)-2]
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        sdist = identies.get(str)
        identies[str] = sdist + dist
        sumdist += dist
        if dist < min_dist:
            min_dist = dist
            identity = name
    print(identies)
    minsum = 100
    for (name,sum) in identies.items():
        if sum/10 < minsum and sum/10 < 0.775:
            minsum = sum/10
            identity = name
        else :
            return None
    '''if sumdist/10 > 0.755:
        print(sumdist / 10)
        return None
    else:
        #print(min_dist)
        print(sumdist/10)
        return str(identity)'''
    print(minsum)
    return str(identity)

def welcome_users(identities):
    global ready_to_detect_identity
    welcome_message = 'Привет '

    if len(identities) == 1:
        welcome_message += '%s, хорошего дня.' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'и %s, ' % identities[-1]
        welcome_message += 'хорошего дня!'

    windows10_voice_interface.Speak(welcome_message)
    ready_to_detect_identity = True

if __name__ == "__main__":
    n = 4
    print("prepare_database", (time.ctime(time.time())))
    #database = prepare_database(n)
    database = {}
    n = 4
    database = load.load(n)
    print("webcam_face_recognizer", (time.ctime(time.time())))
    webcam_face_recognizer(database)
    #photo_face_recognizer(database)
