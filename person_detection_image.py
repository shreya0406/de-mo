import cv2
import numpy as np
import imutils

#generic object detection model
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


def main():
    image = cv2.imread('people.jpg')
    image = imutils.resize(image, width=600)

    #Height and Width of output image file
    (H, W) = image.shape[:2]

    #blob use for infrancing
    blob = cv2.dnn.blobFromImage(image, 0.007843, (W, H), 127.5)
    #pass the blob to the detection
    detector.setInput(blob)
    #All detection result from our model file
    person_detections = detector.forward()
    
    #iterate over all the detection
    for i in np.arange(0, person_detections.shape[2]):
        confidence = person_detections[0, 0, i, 2]
        #checking for threshold value
        if confidence > 0.5:
            idx = int(person_detections[0, 0, i, 1])

            if CLASSES[idx] != "person":
                continue

            person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            #co-ordinates for boundry box of person
            (startX, startY, endX, endY) = person_box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
