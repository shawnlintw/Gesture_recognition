import cv2
import numpy as numpy
from image_module.Skin_segment import Skin_detect
import os

image_folder_path="Train"
result_folder_path="Feature_ext"

class Find_surf(object):

	def __init__(self):
		self.surfDetector = cv2.xfeatures2d.SURF_create()

	def draw(self,image):
		kp,res =  self.surfDetector.detectAndCompute(image,None)
		for marker in kp:
			image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0,255,0))
		return image

def app():
	SURF_feature = Find_surf()
	hand =Skin_detect()

	for root, dirs, files in os.walk(image_folder_path):
		for f in files:
			image = cv2.imread(os.path.join(root,f))
			if image is not None:
				hand_image = hand.Skin_segment(image,'gray')
				feature_image = SURF_feature.draw(hand_image)
			else:
				print('Read Image failed.')
				break
			cv2.imwrite(os.path.join(result_folder_path, 'feature_'+f),feature_image)

if __name__ =='__main__':
	app()
