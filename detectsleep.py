#class to test videos for detecting open and closed eyes.
#Import neccessary libraries.
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

#OpenFace Face recognition library
import openface
import openface.helper
from openface.data import iterImgs
from sklearn.externals import joblib

modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

#Arguement Parser.
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
net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)		#Import the neural net
if args.verbose:
	print("Loading the dlib and OpenFace models took {} seconds.".format(
		time.time() - start))

from imutils import paths;
from facecapture import FaceCapture;

#Class definition begins.
class DetectSleep:

	def __init__(self):
		self.frameSuccess=False;
		self.videoFrame=None;
		self.frameOne=None;
		self.frameTwo=None;
		self.frameThree=None;
		self.meanFrame=None;
		self.closedFrameCount=0;
		self.eyeState=0;

	#Predicts the state of eye as open or closed.
	def getEyeState(self,var,clf):
		res=clf.predict(var);
		return res;

	#Extract features.
	def getRep(self,img):
		img=cv2.resize(img,(96,96));
		rep=net.forwardImage(img);	#Sending frame to neural net.
		if args.verbose:
			print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
			print("Representation:");
			print(rep);
			print("-----\n");
		return rep;

	#Check Blurriness factor.
	def canImageBeConsidered(self,img):
		if img is None:
			print "Not a valid image...";
			return;
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
		fm=cv2.Laplacian(img,cv2.CV_64F).var();		#Extracting the laplacian matrix
		res=True;

		if fm<100:
			res=False;
		
		return res;

	#Generate alert sound in case of emergency
	def alertRequired(self):
		alertTime=1;
		alertFrequency=500;
		os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % ( alertTime, alertFrequency));


	#Eye patch extraction
	def sendFrameToClassifier(self,img,clf):
		if img is None:
			return;
		
		start=time.time();
		bb=align.getLargestFaceBoundingBox(img);

		if bb is None:
			return;
		al=align.align(img,bb);
		
		#wide eye patch
		e_l = al[18][0];
		e_r = al[25][0];
		e_t = al[21][1];
		e_b = al[29][1];

		#narrow eye patch
		narrow_e_l = al[36][0];
		narrow_e_r = al[45][0];
		narrow_e_t = min(al[37][1],al[38][1],al[43][1],al[44][1]); 
		narrow_e_b = max(al[41][1], al[40][1],al[46][1],al[47][1]);
		print np.asarray(bb),"<<<<<<";

		#coordinates of the eyepatch boundaries
		t = bb.top();
		b = bb.bottom();
		l = bb.left();
		r = bb.right();
		print l,r,t,b;

		img1 = img[t:b,l:r];	#Actual patch
		img2 = img[e_t:e_b,e_l:e_r];	#Wide eye patch
		img3 = img[narrow_e_t:narrow_e_b,narrow_e_l:narrow_e_r];	#Narrow eye patch
		d = self.getRep(img2);
		
		#res = clf.predict(d);
		self.eyeState=self.getEyeState(d,clf);
		print self.eyeState;
		var="close";
		self.closedFrameCount+=1;

		if self.eyeState ==1:
		  var = "open";
		  self.closedFrameCount=0;

		#Print to screen
		cv2.putText(img,var, (70,70), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255), 8); 
		cv2.imshow("test video", img);
		if self.closedFrameCount >=5:
			self.alertRequired();

		cv2.waitKey(1);


	#Extract frames from input videos.
	def getFrameFromVideo(self,testVideoPath,testVideoTextFilePath,modelPath):
		vids=os.listdir(testVideoPath);
		videoList=np.loadtxt(testVideoTextFilePath,dtype='str');

		i=0;
		clf=joblib.load(modelPath);		#Load the videos from given path in a list
		for video in videoList:
			print video;
			cap=cv2.VideoCapture(testVideoPath+"/"+video[-1]);
			frameCount=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT));
			print frameCount;
			
			frameIterator=0;
			while(cap.isOpened() and frameIterator<frameCount):
				self.frameSuccess,self.videoFrame=cap.read();

				self.sendFrameToClassifier(self.videoFrame,clf);
				
				frameIterator+=1;
				i+=1;

		cap.release();
		cv2.destroyAllWindows();

	def begin(self,testVideoPath,testVideoTextFilePath,modelPath):
		self.getFrameFromVideo(testVideoPath,testVideoTextFilePath,modelPath);


#FC=FaceCapture();
cam=True;
#cam=FC.isCameraWorking();	#Check if camera is working.

if cam is False:
	print "Camera not working. Aborting.....";
	sys.exit();

DS=DetectSleep();
print "Enter path to folder containing test videos:";
videoPath=raw_input('---->');
print "Enter path to file containing test video names:";
textPath=raw_input('---->');
print "Enter path to model:";
modelPath=raw_input('---->');

DS.begin(videoPath,textPath,modelPath);
