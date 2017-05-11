#class to test videos for detecting open and closed eyes.

import Image;
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



class DetectSleep:

	def __init__(self):
		self.frameSuccess=False;
		self.videoFrame=None;
		self.frameOne=None;
		self.frameTwo=None;
		self.frameThree=None;
		self.meanFrame=None;
		self.eyeState=0;


	def getEyeState(self,var,clf):
		res=clf.predict(var);
		return res;

	def getRep(self,img):
		img=cv2.resize(img,(96,96));
		rep=net.forwardImage(img);
		if args.verbose:
			print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
			print("Representation:");
			print(rep);
			print("-----\n");
		return rep;

	def sendFrameToClassifier(self,img,clf):
		if img is None:
			return;
		if arg.verbose:
			print("  + Original size: {}".format(img.shape));

		start=time.time();
		bb=getLargestFaceBoundingBox(img);

		if bb is None:
			return;
		al=align.align(img,bb);
		
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
		d = getRep(img2);
		
		#res = clf.predict(d);
		self.eyeState=getEyeState(d,clf);
		print self.eyeState;
		var="close";

		
		if self.eyeState ==1:
		  var = "open";


		cv2.putText(img,var, (70,70), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255), 8); 
		cv2.imshow("test video", img);
		if self.eyeState==0:
			alertTime=1;
			alertFrequency=500;
			os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( alertTime, alertFrequency));

		cv2.waitKey(1);

	def getFrameFromVideo(self,testVideoPath,testVideoTextFilePath,modelPath):
		vids=os.listdir(testVideoPath);
		videoList=np.loadtxt(testVideoTextFilePath,dtype='str');

		i=0;
		clf=joblib.load(modelPath);
		for video in videoList:
			print video;
			cap=cv2.VideoCapture(testVideoPath+"/"+video[-1]);
			frameCount=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
			print frameCount;
			
			frameIterator=0;
			while(cap.isOpened() and frameIterator<frameCount):
				self.frameSuccess,self.videoFrame=cap.read();

				sendFrameToClassifier(self.videoFrame,clf);
				
				frameIterator+=1;
				i+=1;

		cap.release();
		cv2.destroyAllWindows();

	def begin(self,testVideoPath,testVideoTextFilePath,modelPath):
		getFrameFromVideo(self,testVideoPath,testVideoTextFilePath,modelPath);


DS=DetectSleep();
print "Enter path to folder containing test videos:";
videoPath=raw_input('---->');
print "Enter path to file containing test video names:";
textPath=raw_input('---->');
print "Enter path to model:";
modelPath=raw_input('---->');

DS.begin(videoPath,textPath,modelPath);
