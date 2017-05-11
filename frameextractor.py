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

#Importing openface and other neccessary libraries.
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

if args.verbose:
	print("Loading the dlib and OpenFace models took {} seconds.".format(
		time.time() - start))



class FrameExtractor:

	def __init__(self):
		self.success=False;
		self.frameCount=0;
		self.videoFilePath="";
		#self.videoFileName=[];

	#Resets frameCount to 0
	def resetFrameCount(self):
		self.frameCount=0;

	#Set videoFileName to arguement path which specifies the location of training videos.
	def setVideoFileName(self,path):
		self.videoFilePath=path;

	#Returns name of files which are to be processed
	def getVideoFileName(self,eyeStatePath):
		if not os.path.exists(eyeStatePath):
			print "Not a proper location of eye_state file.";
			return;

		videos=[];
                txt =np.loadtxt(eyeStatePath, dtype='str')
                
		for stf in txt:
		
			videos.append(stf[-1]);

		return videos;

	#Extract wide and narrow eye patches from video frame and stores it.
	def createPatch(self,path_wide,path_narrow,img):

		if img is None:
                        return

		start = time.time();
		#getLargestFaceBounding : responsible for extracting the facial region from image.
		bb = align.getLargestFaceBoundingBox(img);
		if bb is None:
			return;
		al = align.align(img,bb);
	
		#Wide eye patches
		e_l = al[18][0]; #Left position of eye region
		e_r = al[25][0]; #Right position
		e_t = al[21][1]; #Top position
		e_b = al[29][1]; #Bottom position
	
		#Narrow Eye patches.
		narrow_e_l = al[36][0];	#left
		narrow_e_r = al[45][0];	#right
		narrow_e_t = min(al[37][1],al[38][1],al[43][1],al[44][1]);	#top 
		narrow_e_b = max(al[41][1], al[40][1],al[46][1],al[47][1]);	#bottom
	
		print np.asarray(bb),"<<<<<<";
	
		t = bb.top();
		b = bb.bottom();
		l = bb.left();
		r = bb.right();
	
		print l,r,t,b;
	
		img1 = img[t:b,l:r]; #Facial Area
		img2 = img[e_t:e_b,e_l:e_r];	#Wide eye patch region
		img3 = img[narrow_e_t:narrow_e_b,narrow_e_l:narrow_e_r];	#Narrow eye patch region
		a,b,c = img1.shape;
	
		print a,b,c; #printing..
	
		#Resizing all the image to the same size for concatenation
		rs_img = cv2.resize(img, (b,a));
		rs_img2 = cv2.resize(img2, (b,a));
		rs_img3 = cv2.resize(img3, (b,a));
		
		#Concatenating the actual image,facial porion, wide eye patch, narrow patch.
		vis = np.concatenate((rs_img, img1), axis=1);
		vis2 = np.concatenate((vis, rs_img2), axis=1);
		vis3 = np.concatenate((vis2, rs_img3), axis=1);
	
		
		print path_wide+str(self.frameCount)+".jpg"
		
		#Storing the eye patches.
		cv2.imwrite(path_wide+str(self.frameCount)+".jpg",img2);
		cv2.imwrite(path_narrow+str(self.frameCount)+".jpg",img3);

		#visualizing the eye patches.
		cv2.imshow('visualization', vis3);
		cv2.waitKey(1);
		

	#Processing videos, extracting frames.
	def fetchFrame(self,eyeStatePath,eyeState):
		self.videoFileName=self.getVideoFileName(eyeStatePath);
		print self.videoFileName

		#create directories where eye patches will be stored.
		if not os.path.exists(self.videoFilePath+"/"+"Frames_"+str(eyeState)+"_wide_eye_patch/"):
			os.makedirs(self.videoFilePath+"/Frames_"+ str(eyeState)+"_wide_eye_patch/");

		if not os.path.exists(self.videoFilePath+"/"+"Frames_"+str(eyeState)+"_narrow_eye_patch/"):
			os.makedirs(self.videoFilePath+"/Frames_"+ str(eyeState)+"_narrow_eye_patch/");

		#Looping through all the videos.
		for file in self.videoFileName:
			try:
				curVideo=cv2.VideoCapture(self.videoFilePath+"/"+file); #Opening a video for processing.
			except:
				print "Cannot read "+file;continue;

			try:
				self.success,self.videoFrame=curVideo.read(); #Reading first frame.
			except:
				print "Cannot read frames of "+file;continue;

			self.success=True;
			while self.success:
				self.success,self.videoFrame=curVideo.read();

				path_wide=self.videoFilePath+"/Frames_"+str(eyeState)+"_wide_eye_patch/";	#path to folder where wide eye patches will be stored.
				path_narrow=self.videoFilePath+"/Frames_"+str(eyeState)+"_narrow_eye_patch/";	#path to folder where narrow eye patches will be stored.
				self.createPatch(path_wide,path_narrow,self.videoFrame);
				self.frameCount+=1;

			print "Frames of "+file+" fetched successfully.";

			



#FE=FrameExtractor();
#FE.setVideoFileName("/home/priyanjit/Codes");
#FE.fetchFrame();
####
