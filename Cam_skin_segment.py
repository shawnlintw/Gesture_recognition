import cv2
from image_module.Skin_segment import Skin_detect


if __name__ =='__main__':
	cap =cv2.VideoCapture(0)
	hand = Skin_detect()

	while (cap.isOpened()):
		ret,frame=cap.read()
		frame = cv2.flip(frame,1)
		
		roi = frame[100:500, 700:1200]
		cv2.rectangle(frame, (700,100),(1200,500),(0,0,0),4)

		HSV_mask, YCbCr_mask, hand_image = hand.Skin_segment(roi,'show_mask')
		
		cv2.imshow('Camera', frame)
		cv2.imshow('HSV_mask',HSV_mask)
		cv2.imshow('YCbCr_mask',YCbCr_mask)
		cv2.imshow('Result',hand_image)
		k = cv2.waitKey(1)&0xff	
		if k==27:
			break
	cap.release()
	cv2.destroyAllWindows()