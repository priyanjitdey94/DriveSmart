#Class responsible for extracting frames from videos

import glob;
import time

start = time.time()
import argparse
import cv2
import itertools
import os
from sklearn import svm
import numpy as np
np.set_printoptions(precision=2)

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface
import openface.helper
from openface.data import iterImgs
from sklearn.externals import joblib

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",default=os.path.join(dlibModelDir, "mean.csv"))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--dlibRoot', type=str,default=os.path.expanduser("~/src/dlib-18.16/python_examples"),help="dlib directory with the dlib.so Python library.")
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
parser.add_argument('--imgDim', type=int,help="Default image dimension.", default=96)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()

sys.path.append(args.dlibRoot)
import dlib

from openface.alignment import NaiveDlib  # Depends on dlib.
if args.verbose:
	print("Argument parsing and loading libraries took {} seconds.".format(
		time.time() - start))

start = time.time()
align = NaiveDlib(args.dlibFaceMean, args.dlibFacePredictor)
net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
if args.verbose:
	print("Loading the dlib and OpenFace models took {} seconds.".format(
		time.time() - start))
class FrameExtractor:

	def __init__(self):
		self.success=False;
		self.frameCount=0;
		self.videoFilePath="";
		#self.videoFileName=[];

	def resetFrameCount(self):
		self.frameCount=0;

	def setVideoFileName(self,path):
		self.videoFilePath=path;

	def getVideoFileName(self,eyeStatePath):
		if not os.path.exists(eyeStatePath):
			print "Not a proper location of eye_state file.";
			return;

		videos=[];
		for file in open(eyeStatePath):
			file.rstrip('\n');
			videos.append(file);

		return videos;

	def createPatch(self,path_wide,path_narrow,img):
		if img is None:
			print "Couldn't open "+file+". Moving to next one.";continue;
		if args.verbose:
			print("  + Original size: {}".format(img.shape));

		start = time.time();
		bb = align.getLargestFaceBoundingBox(img);
		if bb is None:
			return;
		al = align.align(img,bb);
	
		e_l = al[18][0];
		e_r = al[25][0];
		e_t = al[21][1];
		e_b = al[29][1];
	
		narrow_e_l = al[36][0];
		narrow_e_r = al[45][0];
		narrow_e_t = min(al[37][1],al[38][1],al[43][1],al[44][1]); 
		narrow_e_b = max(al[41][1], al[40][1],al[46][1],al[47][1]);
	
		print np.asarray(bb),"<<<<<<";
	
		t = bb.top();
		b = bb.bottom();
		l = bb.left();
		r = bb.right();
	
		print l,r,t,b;
	
		img1 = img[t:b,l:r];
		img2 = img[e_t:e_b,e_l:e_r];
		img3 = img[narrow_e_t:narrow_e_b,narrow_e_l:narrow_e_r];
		a,b,c = img1.shape;
	
		print a,b,c;
	
		rs_img = cv2.resize(img, (b,a));
		rs_img2 = cv2.resize(img2, (b,a));
		rs_img3 = cv2.resize(img3, (b,a));
	
		vis = np.concatenate((rs_img, img1), axis=1);
		vis2 = np.concatenate((vis, rs_img2), axis=1);
		vis3 = np.concatenate((vis2, rs_img3), axis=1);
	
		#cv2.imwrite(path+"_eye_patch/"+str(self.frameCount)+".jpg",img2);
		#cv2.imwrite(path+"_eye_patch/"+str(self.frameCount)+".jpg",img3);
		cv2.imwrite(path_wide+str(self.frameCount)+".jpg",img2);
		cv2.imwrite(path_narrow+str(self.frameCount)+".jpg",img3);
		cv2.imshow('visualization', vis3);
		cv2.waitKey(1);
		#num+=1;

	def fetchFrame(self,eyeStatePath,eyeState):
		self.videoFileName=self.getVideoFileName(eyeStatePath);
		
		if not os.path.exists(self.videoFilePath+"/"+"Frames_"+str(eyeState)+"_wide_eye_patch/"):
			os.makedirs(self.videoFilePath+"/Frames_"+ str(eyeState)+"_wide_eye_patch/");

		if not os.path.exists(self.videoFilePath+"/"+"Frames_"+str(eyeState)+"_narrow_eye_patch/"):
			os.makedirs(self.videoFilePath+"/Frames_"+ str(eyeState)+"_narrow_eye_patch/");

		for file in self.videoFileName:
			try:
				curVideo=cv2.VideoCapture(self.videoFilePath+"/"+file);
			except:
				print "Cannot read "+file;continue;

			try:
				self.success,self.videoFrame=curVideo.read();
			except:
				print "Cannot read frames of "+file;continue;

			self.success=True;
			while self.success:
				self.success,self.videoFrame=curVideo.read();

				#f_img=self.videoFilePath+"/Frames_"+str(eyeState)+str(self.frameCount)+".jpg";
				#cv2.imwrite(f_img,self.videoFrame);
				path_wide=self.videoFilePath+"/Frames_"+str(eyeState)+"_wide_eye_patch/";
				path_narrow=self.videoFilePath+"/Frames_"+str(eyeState)+"_narrow_eye_patch/";
				self.createPatch(path_wide,path_narrow,self.videoFrame);
				self.frameCount+=1;

			print "Frames of "+file+" fetched successfully.";

			



#FE=FrameExtractor();
#FE.setVideoFileName("/home/priyanjit/Codes");
#FE.fetchFrame();
####
