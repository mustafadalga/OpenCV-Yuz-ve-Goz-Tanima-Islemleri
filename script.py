#**************************************#
__author__ = "Mustafa Dalga"
__email__ = "mustafadalgaa@gmail.com"
__opencv__version__="4.0.0"
#**************************************#

import cv2
kamera=cv2.VideoCapture(0)
yuz_casc=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
goz_casc=cv2.CascadeClassifier("haarcascade_eye.xml")


while True:
	_,goruntu=kamera.read()

	griTon=cv2.cvtColor(goruntu,cv2.COLOR_BGR2GRAY)
	yuzler=yuz_casc.detectMultiScale(griTon,1.3,5)
	for (x,y,w,h) in yuzler:
		cv2.rectangle(goruntu,(x,y),(x+w,y+h),(0,255,0),3)
		roi_griTon=griTon[y:y+h,x:x+w]
		roi_renkli=goruntu[y:y+h,x:x+w]
		gozler=goz_casc.detectMultiScale(roi_griTon)
		for (ex,ey,ew,eh) in gozler:
			cv2.rectangle(roi_renkli,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)

	cv2.imshow("Orjinal",goruntu)
	if cv2.waitKey(1)==ord("q"):
		break;

kamera.release()
cv2.destroyAllWindows()
