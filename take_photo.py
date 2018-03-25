import cv2
cam = cv2.VideoCapture(0)
name="pranoy"
cv2.namedWindow("test")
from align import AlignDlib

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


alignment = AlignDlib('models/landmarks.dat')

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "images/"+name+"/"+name+str(img_counter)+".jpg"
        if align_image(frame) is not None:

            cv2.imwrite(img_name, frame)

            print("{} written!".format(img_name))
            img_counter += 1
        else:
            print("No Face Found")

cam.release()

cv2.destroyAllWindows()