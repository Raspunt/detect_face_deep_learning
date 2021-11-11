from imutils.video import VideoStream
import imutils
import numpy as np
import cv2
import time

vs = VideoStream(src=0).start() 

prototxt   = "dataSet/deploy.prototxt.txt"
model      = "dataSet/res10_300x300_ssd_iter_140000.caffemodel"
confidenceArgs = 0.5


time.sleep(2)


net = cv2.dnn.readNetFromCaffe(prototxt, model)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=1000, height=1000)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

    net.setInput(blob)
    detections = net.forward()


    for i in range(0,detections.shape[2]):

        confidence =detections[0,0,i,2]  

        if confidence < confidenceArgs:
            continue
    
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])

        (startX,startY,endX,endY) = box.astype("int")

        text = f"{confidence * 100}"

        y = startY - 10 
        

        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,0,255),2)

        cv2.putText(frame,text,(startX,y),
            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

    cv2.imshow("Face detector  from camera",frame)
    key = cv2.waitKey(1) &  0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()