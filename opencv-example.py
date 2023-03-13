import cv2
import imutils

#read the image
#image = cv2.imread('input_image.JPG')
cap = cv2.VideoCapture('test_video.mp4')

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)

    text = "This is my custom text"
    cv2.putText(frame, text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255), 1)
    #(image , pass text, pass the origin(X, Y coordinates),font,size of text,RGB, Thickness)

    cv2.rectangle(frame, (50,50), (500 , 500), (0, 0, 255), 2)
    #(image, co-ordinates of 1st point, point 2 co-ordinates, RGB,Thickness)
    cv2.imshow('img', frame)
    
    #to save the image use imwrite()
    #cv2.imwrite('output_image.jpg', image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()


