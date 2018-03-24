import pickle as pk
import cv2
import dlib
from align import AlignDlib
from load_data import load_metadata
import numpy as np
from model import create_model


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def align_image(img):
    return alignment.align(96, img, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

def encoding(img):
    img = align_image(img)
    # scale RGB values to interval [0,1]
    cv2.imwrite("a.jpg",img)
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    return embedded


def distance(emb1, emb2):
    #return np.linalg.norm( np.subtract(emb1, emb2) )
    return np.sum(np.square(emb1 - emb2))


camera_port = 0
camera = cv2.VideoCapture(camera_port)

database = pk.load(open("database.pkl", "rb"))

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

result=0.3
min_threshold = 0.3            #Consider this as the standard threshold value
predict_name = "Unknown"

while True:

    #Reading image from webcam
    retval, image = camera.read()

    #Drawing bounding box
    temp_image = image

    bb = alignment.getLargestFaceBoundingBox(temp_image)
    if (bb is not None):
        cv2.rectangle(temp_image, (bb.left(), bb.top()) ,(bb.right(), bb.bottom()), (0,255,0),1)
        cv2.putText(temp_image, predict_name+" : "+str(result)+"%", (bb.left(),bb.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

    cv2.imshow(predict_name, temp_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    try :
        # turn and crop the face
        image = align_image(image)        # image.shape = (96, 96, 3)

        # scale RGB values to interval [0,1]
        image = (image / 255.).astype(np.float32)

        # obtain embedding vector for image
        encoded_image = nn4_small2_pretrained.predict(np.expand_dims(image, axis=0))[0]

        for (name,db_enc) in database.items():

            result=distance(db_enc,encoded_image)

            if(result < min_threshold):

                #min_threshold = result
                predict_name = name

                print(result)
                print(name)
                break

    except:
        print ("Face not detected !")

camera.release()
cv2.destroyAllWindows()