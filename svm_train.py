import cv2
from image_module.bow_svm import BOW 
from image_module.Skin_segment import Skin_detect


if __name__=='__main__':
	
	
	train_folder_path = 'Train'
	bow_num=300
	VOC_filename='Svm_xml/voc_least.npy'
	SVMXml_filename ='Svm_xml/svm_least.xml'

	hand_img=Skin_detect()
	bow=BOW()

	print("SVM Training start")
	bow.trainSVM(train_folder_path,hand_img.Skin_segment,'gray',bow_num,VOC_filename,SVMXml_filename)
	print("SVM Training finished.")
	
	print("VOC save on :"+VOC_filename +" and SVM save on:" + SVMXml_filename)

	test_file = "Test/test.png"
	img = cv2.imread(test_file)
	
	result=bow.predict(hand_img.Skin_segment(img,'gray'),VOC_filename,SVMXml_filename)
	print(test_file + ' is :'+ str(result))
