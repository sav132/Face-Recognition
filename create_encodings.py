import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from load_data import load_metadata
import numpy as np
from model import create_model
from align import AlignDlib
from pickle import  dump


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
metadata=load_metadata("images")
alignment = AlignDlib('models/landmarks.dat')

def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


database = {}
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    database.update({m.image_path().split("/")[1]:nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]})

dump(database, open("database.pkl","wb"))