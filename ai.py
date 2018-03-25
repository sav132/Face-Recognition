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

def dbEncodings_to_Vect(database):
    db_vect = []
    names = []
    for (name,db_enc) in database.items():
        names.append(name)
        db_vect.append(db_enc)
    db_vect = np.array(db_vect).T       # db_vect.shape = (128, len(db))
    return db_vect, names


camera_port = 0
camera = cv2.VideoCapture(camera_port)

# Open database
database = pk.load(open("database.pkl", "rb"))
# Make vectors
db_vect, names = dbEncodings_to_Vect(database)

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

result = 0
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

    cv2.imshow("Window", temp_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    try :

        # 1. IMAGE --> EMBEDDING
        # Turn and crop the face
        image = align_image(image)        # image.shape = (96, 96, 3)
        # Scale RGB values to interval [0,1]
        image = (image / 255.).astype(np.float32)
        # Obtain the embedding vector for image
        encoded_image = nn4_small2_pretrained.predict(np.expand_dims(image, axis=0))[0]


        # 2.EMBEDDING --> DISTANCES
        encoded_image = np.reshape(encoded_image, (128,1))
        # Computing Dictances
        dist_vect = np.sum(np.square(np.subtract(db_vect, encoded_image)), axis=0, dtype=np.float32, keepdims=True)
        dist_vect = np.squeeze(dist_vect)


        # 3.DISTANCES --> NAME
        result = dist_vect.min()
        if(result < 0.2):
            #threshold = result
            predict_name = names [ list(dist_vect).index(result) ]
            print(predict_name, dist_vect.min())
        else:
            predict_name = "Unknown"

    except:
        print ("Face not detected !")

camera.release()
cv2.destroyAllWindows()