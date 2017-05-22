#Class responsible for SVM training

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

if args.verbose:
	print("Loading the dlib and OpenFace models took {} seconds.".format(
		time.time() - start))
######################################################################################################

#Class definition for training model using SVM.
class TrainModel:

	def __init__(self):
		self.modelName="defaultName";
		self.modelLocation="/home/trainedModel";
		self.modelFormat="pkl";

	def getModelName(self):
		return self.modelName;

	def setModelName(self,name):
		self.modelName=name;

	def getModelLocation(self):
		return self.modelLocation;

	def setModelLocation(self,path):
		self.modelLocation=path;

	def getModelFormat(self):
		return self.modelFormat;

	def setModelFormat(self,format):
		self.modelFormat=format;

	#Creating a SVM classifier and feeding the training data
	def startTraining(self):
		feat_x = np.load('/home/prithviraj/open_feature_new/of.npy');
		feat_y= np.load('/home/prithviraj/open_feature_new/oflabel.npy');
		print feat_x.shape, feat_y.shape;
		print "Training...........";

		#defining a SVM classifier
		clf = svm.SVC();
		clf.fit(feat_x, feat_y);	#feeding data to SVM
		joblib.dump(clf, '/home/prithviraj/open_feature_new/trained_svm.pkl',compress=1);
		print "Training Complete.";


#import required classes.
from frameextractor import FrameExtractor;
from labelframe import LabelFrame;

#create objects.
FE=FrameExtractor();
LF=LabelFrame();
TM=TrainModel();

print "====== Welcome to DriveSmart Training Module ======";
print "Enter the path to the video folder:";
videoPath=raw_input('--->');
print "Enter the text file containing list of videos with closed eyes:";
closedTextPath=raw_input('--->');
print "Enter the text file containing list of videos with open eyes:";
openTextPath=raw_input('--->');

#Extracting Frames 
FE.setVideoFileName(videoPath);
FE.fetchFrame(closedTextPath,0);
FE.fetchFrame(openTextPath,1);

#Extracting Eye Patches
#LF.getEyePatch(videoPath+"/Frames_0");
#LF.getEyePatch(videoPath+"/Frames_1");

#Labelling Eye Patches
LF.labelFrame(videoPath);

#Training Model
TM.startTraining();