import cv2
import numpy as np

cap = cv2.VideoCapture('test.mp4')


while True:
  success, image = cap.read()
  if success:
    imgContour = image.copy()
    imgcrop=image[0:100,295:560]
    cropCopy=imgcrop.copy()
    imgGray=cv2.cvtColor(imgcrop,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny=cv2.Canny(imgBlur,50,100)
    contours, _ =cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        print(area)
        if area>2000:
            peri=cv2.arcLength(cnt,True)
            print(peri)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            points=len(approx)
            print(points)
            x,y,w,h=cv2.boundingRect(approx)
            if points>4:
                name="circle"
                cv2.rectangle(cropCopy,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.putText(cropCopy,"circle",(x,y+60),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(image,(0,220),(856,300),(0,0,255),cv2.FILLED)
                cv2.putText(image,"FAULTY ITEM DETECTED",(90,280),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
            else:
                cv2.rectangle(cropCopy,(x,y),(x+w,y+h),(0,255,0),1)
                cv2.putText(cropCopy,"Rectangle",(x,y+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)    
    cv2.imshow("threshold",imgCanny)
    cv2.imshow("result",image)
    cv2.imshow("crop",cropCopy)
    
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
    
  else:
      break