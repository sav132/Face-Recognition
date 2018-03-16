import cv2
import os


folder="pranoy"


camera_port = 0

ramp_frames = 30

camera = cv2.VideoCapture(camera_port)

l=5
i=1

while True :

    retval, image = camera.read()

    #photo=pygame.image.save(img, 'dataset_image/1/image'+str(i)+'.jpg')

    try:

            if(i<=l):
                cv2.imwrite('images/'+folder+'/'+folder+str(i)+'.jpg', image)
                i=i+1
                print("image detected")
            else:
                break
    except:
        print('1')


