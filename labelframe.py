#Class responsible for labeling eye patches as closed/open

import Image;
import glob;
import time

start = time.time()
import argparse
import imutils
import cv2
import itertools
import os
from sklearn import svm
import numpy as np
np.set_printoptions(precision=2)

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

#import openface and other neccessary libraries.
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

#Importing torch neural net
net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
if args.verbose:
	print("Loading the dlib and OpenFace models took {} seconds.".format(
		time.time() - start))

class LabelFrame:

	def __init__(self):
		self.feat_x=[];
		self.feat_y=[];

	#Decide if a frame is hazy
	def isValidFrame(self,imagePath):
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		fm = variance_of_laplacian(gray)
		text = "Not Blurry"

		if fm<int(100):
			text="Blurry"

		if text=="Blurry":
			return False;
		else:
			return True;

	#def setFrame(self):

	#def getFrame(self):

	def getEyePatch(self,path):
		num=0;
		if not os.path.exists(path+"_eye_patch"):
			os.mkdir(path+"_eye_patch");

		try:
			os.chdir(path);
		except:
			print "Invalid location. Can't find frames.";
			print "Aborting at labelling stage...........";
			return;

		for file in glob.glob("*.jpg"):
			img=Image.open(path+"/"+file);
			if img is None:
				print "Couldn't open "+file+". Moving to next one.";continue;
			if args.verbose:
				print("  + Original size: {}".format(img.shape));

			start = time.time();
			#getLargestFaceBounding : responsible for extracting the facial region from image.
			bb = align.getLargestFaceBoundingBox(img);
			if bb is None:
				return;
			al = align.align(img,bb);
			
			#wide eye patches
			e_l = al[18][0];
			e_r = al[25][0];
			e_t = al[21][1];
			e_b = al[29][1];
			
			#narrow eye patches
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
			
			#Resizing all the image to the same size for concatenation
			rs_img = cv2.resize(img, (b,a));
			rs_img2 = cv2.resize(img2, (b,a));
			rs_img3 = cv2.resize(img3, (b,a));
		
			#Concatenating the actual image,facial porion, wide eye patch, narrow patch.
			vis = np.concatenate((rs_img, img1), axis=1);
			vis2 = np.concatenate((vis, rs_img2), axis=1);
			vis3 = np.concatenate((vis2, rs_img3), axis=1);
		
			#Storing the eye patches.
			cv2.imwrite(path+"_eye_patch/"+str(num)+".jpg",img2);
			cv2.imwrite(path+"_eye_patch/"+str(num)+".jpg",img3);
			
			#visualizing the eye patches.
			cv2.imshow('visualization', vis3);
			cv2.waitKey(1);
			num+=1;

		print "Eye patch extraction successful.Moving to labelling....";


	#Representation of 128 dimensional features.
	def getRep(self, imgPath):
		if args.verbose:
			print("Processing {}.".format(imgPath))
		img = cv2.imread(imgPath)
		
		img = cv2.resize(img,(96,96))
		rep = net.forwardImage(img)
		if args.verbose:
			print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
			print("Representation:")
			print(rep)
			print("-----\n")
		return rep


	#Processing videos.
	def labelFrame(self,path):
		#Looping through wide eye patches
		for img in os.listdir(path+"/Frames_0_wide_eye_patch"):
                        #print img, "<<"
			d = self.getRep(path+"/Frames_0_wide_eye_patch/"+img)
			print d.shape
			print "Appending..", img
			self.feat_x.append(d)
			self.feat_y.append(0);

		#Looping through narrow eye patches.
		for img in os.listdir(path+"/Frames_1_wide_eye_patch"):
			d = self.getRep(path+"/Frames_1_wide_eye_patch/"+img)
			print d.shape
			print "Appending..", img
			self.feat_x.append(d)
			self.feat_y.append(1);

		print "Storing............"
		os.mkdir('/home/prithviraj/open_feature_new')
		np.save('/home/prithviraj/open_feature_new/of.npy', self.feat_x)
		np.save('/home/prithviraj/open_feature_new/oflabel.npy', self.feat_y)
