import numpy as np
import cv2
import os

class BOW(object):

	def __init__(self):
		self.feature_detector = cv2.xfeatures2d.SURF_create(500)
		self.descript_extractor = cv2.xfeatures2d.SURF_create()
		self.descript_extractor.setExtended(True)

		flann_params = dict(algorithm=1,tree=5)
		flann = cv2.FlannBasedMatcher(flann_params,{})
		self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descript_extractor,flann)

	def SURF_descriptor_extractor(self,img):
		return self.descript_extractor.compute(img,self.feature_detector.detect(img))[1] 		
		
	def BOW_descriptor_extractor(self,img):
		feature_des = self.feature_detector.detect(img)
		if feature_des is None:
			return None
		else:
			return self.bow_img_descriptor_extractor.compute(img,feature_des)

	def genVOC(self,bow_num,train_folder_path,img_process_mothod,img_process_arg):
		bow_trainer = cv2.BOWKMeansTrainer(bow_num)
		dirs=os.listdir(train_folder_path)

		for Dir in dirs :
			files = os.listdir(os.path.join(train_folder_path,Dir))
			file_count=0
			for ImageFile_name in files:
				file = cv2.imread(os.path.join(train_folder_path,Dir,ImageFile_name))
				file_count +=1
				if img_process_mothod !=0:
					roi = img_process_mothod(file,img_process_arg)
				else:
					continue 
				kmeans_obj = self.SURF_descriptor_extractor(roi)
				if kmeans_obj is None:
					continue
				bow_trainer.add(kmeans_obj)
				if file_count==20:
					break
		return bow_trainer.cluster()
		
	def genSVM_TRAINDATA(self,train_folder_path,img_process_mothod,img_process_arg):
		traindata, trainlabels =[], []
		dirs = os.listdir(train_folder_path)
		for Dir in dirs:
			files = os.listdir(os.path.join(train_folder_path,Dir))
			for ImageFile_name in files:
				file = cv2.imread(os.path.join(train_folder_path,Dir,ImageFile_name))
				roi= img_process_mothod(file,img_process_arg)
				if roi is None:
					continue
				train_obj = self.BOW_descriptor_extractor(roi)
				if train_obj is None:
					continue
				traindata.extend(train_obj)
				if Dir == 10:
					trainlabels.append(-1)
				else:
					trainlabels.append(int(Dir))
		return traindata , trainlabels


	def trainSVM(self,train_folder_path,img_process_mothod,img_process_arg,bow_num,VOC_filename,SVMXml_filename):
		
		voc = self.genVOC(bow_num,train_folder_path,img_process_mothod,img_process_arg)
		
		np.save(VOC_filename,voc)	#Save the VOC file to VOC_filename.npy
		print('bow training finished.')
		self.bow_img_descriptor_extractor.setVocabulary(voc)

		traindata, trainlabels = self.genSVM_TRAINDATA(train_folder_path,img_process_mothod,img_process_arg)

		self.svm = cv2.ml.SVM_create()
		self.svm.setType(cv2.ml.SVM_C_SVC)
		self.svm.setKernel(cv2.ml.SVM_RBF)
		self.svm.setC(0.1)
		self.svm.setGamma(0.01)
		self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
		self.svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
		self.svm.save(SVMXml_filename)
		print("svm training finished.")

	def predict(self,img,VOC_filename,SVMXml_filename):
		voc = np.load(VOC_filename)
		self.bow_img_descriptor_extractor.setVocabulary(voc)
		data = self.BOW_descriptor_extractor(img)
		
		self.svm=cv2.ml.SVM_load(SVMXml_filename)

		if data is None:
			return 10
		else:
			res=self.svm.predict(data)
			return int(res[1][0][0])
