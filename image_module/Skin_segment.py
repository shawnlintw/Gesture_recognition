import cv2
import numpy as np

class Skin_detect(object):

	def Skin_HSV(self,image):
		image_HSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		HSV_mask = cv2.inRange(image_HSV, (0,15,0),(150,255,255))
		HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
		HSV_result = cv2.bitwise_not(HSV_mask)

		return HSV_mask, HSV_result
		
	def Skin_YCrCb(self,image):
		image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
		YCrCb_mask = cv2.inRange(image_YCrCb,(0,135,85),(255,190,135))
		YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((10,10), np.uint8))
		YCrCb_result = cv2.bitwise_not(YCrCb_mask)
		
		return YCrCb_mask, YCrCb_result


	def Skin_segment(self,image,type):
		# HSV color space
		HSV_mask, HSV_result = self.Skin_HSV(image)
		# YCrCb color space
		YCrCb_mask, YCrCb_result = self.Skin_YCrCb(image)

		# Combination there HSV and YCbCr methods.
		result_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
		result_mask = cv2.medianBlur(result_mask,3)
		result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, np.ones((4,4),np.uint8))
		segment_color = cv2.bitwise_and(image, image, mask = result_mask)

		if type =='color' :
			return segment_color
		elif type == 'gray' :
			segment_gray = cv2.cvtColor(segment_color, cv2.COLOR_BGR2GRAY)
			segment_gray = cv2.equalizeHist(segment_gray)
			return segment_gray
		elif type == 'show_mask' :
			return HSV_mask, YCrCb_mask, segment_color
		else:
			return segment_color


def app():
	image = cv2.imread("../Test/test.png")
	Hand = Skin_detect()
	hand_image = Hand.Skin_segment(image,'color')
	cv2.imshow('Hand',hand_image)
	while True:
		k = cv2.waitKey(0)&0xff
		if k==27:
			break
	cv2.destroyAllWindows()


if __name__ =='__main__':
	app()
