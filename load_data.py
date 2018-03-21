import numpy as np
import os.path
import numpy as np
from align import AlignDlib
import random
import cv2

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')


def pre_processing(image):
    # Detect face and return bounding box
    bb = alignment.getLargestFaceBoundingBox(image)

    # Transform image using specified face landmark indices and crop image to 96x96
    aligned = alignment.align(96, image, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

def generator():
    dataset=load_images("images")

    anchor=[]
    p_img=[]
    n_img=[]
    for  i,j,k in dataset:

        anchor.append(i)
        p_img.append(j)
        n_img.append(k)

    anchor=np.array(anchor)
    p_img=np.array(p_img)
    n_img=np.array(n_img)
    return anchor,p_img,n_img


def load_images(path):
    dataset=[]
    labels = os.listdir(path)
    for label in labels:

        images = os.listdir(path+"/"+label)

        for image in images:
            anchor=cv2.imread(path+"/"+label+"/"+image)
            anchor = pre_processing(anchor)
            for image1 in images:
                p_img = cv2.imread(path+"/"+label+"/"+image1)
                p_img = pre_processing(p_img)
                labels1=[i for i in labels if i!=label]
                for label1 in labels1:

                        images2 = os.listdir(path+"/"+label1)
                        for image2 in images2:

                            n_img = cv2.imread(path+"/"+label1+"/"+image2)
                            n_img=pre_processing(n_img)
                            dataset.append((anchor,p_img,n_img))
    return dataset


