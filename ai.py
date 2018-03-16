from pickle import load
import cv2
from align import AlignDlib
from load_data import load_metadata
import numpy as np
from model import create_model

camera_port = 0
camera = cv2.VideoCapture(camera_port)


database = load(open("database.pkl", "rb"))

nn4_small2 = create_model()
# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)



def encoding(img):

    img = align_image(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    return embedded


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(enc1, enc2):

    result=distance(enc1,enc2)
    return result


while True:
  try:
    retval, image = camera.read()
    #image=cv2.imread("rejah14.jpg")
    encoded_image=encoding(image)
    for (name,db_enc) in database.items():
        result=show_pair(db_enc,encoded_image)
        if(result<0.5):
           print(result)
           print(name)
           break

  except:
      print("1")

