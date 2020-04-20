import cv2
import os
from image_module.bow_svm import BOW 
from image_module.Skin_segment import Skin_detect

if __name__ =='__main__':
	
	VOC_filename = 'SVM_xml/voc_least.npy'
	test = 'SVM_xml/svm_least.xml'

	hand = Skin_detect()
	bow = BOW()

	cam=cv2.VideoCapture(0)
	while True:
		ret,frame=cam.read()
		frame = cv2.flip(frame,1)
		roi =frame[100:500, 700:1200]
		cv2.rectangle(frame, (700,100),(1200,500),(0,255,0),4)
		
		gray = hand.Skin_segment(roi,'gray')

		result =bow.predict(gray,VOC_filename,test)
		text = str(result)
		if result == 10:
			text = 'None'
		cv2.putText(frame,text,(700,80),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(128,255,128),3,cv2.LINE_AA)
		cv2.imshow('Gesture',frame)
		#cv2.imshow('roi',roi)
		cv2.imshow('gray',gray)
		k = cv2.waitKey(1) & 0xff
		if k ==27:
			break

	cv2.destroyAllWindows()
