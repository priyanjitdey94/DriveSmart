#Class responsible for labeling eye patches as closed/open

import Image;
import glob;
import sys;
import os;
import cv2;
import time;
import argparse;
import itertools;
import numpy as np;

np.set_printoptions(precision=2)
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface
import openface.helper
from openface.data import iterImgs

# Path to include pre-trained models
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

# Path to dlib face predictor
parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "mean.csv"))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--dlibRoot', type=str,
                    default=os.path.expanduser(
                        "~/src/dlib-18.16/python_examples"),
                    help="dlib directory with the dlib.so Python library.")

# Path to torch neural net
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))

# Specify image dimension. 96 in this case.
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)

# Specify whether to use CUDA
parser.add_argument('--cuda', action='store_true')

#Specify whether to use Verbose
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


class LabelFrame:


	#def isValidFrame(self):

	#def setFrame(self):

	#def getFrame(self):

	def labelFrame(self,path):
		counter=0;
		try:
			os.chdir(path);
		except:
			print "Invalid location. Can't find frames.";
			print "Aborting at labelling stage...........";
			return;

		for file in glob.glob("*.jpg"):
			img=Image.open(path+file);
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
		    
		    cv2.imwrite('/home/prithviraj/closed_eyes/c_patch_'+str(num)+'.jpg',img2);
		    cv2.imwrite('/home/prithviraj/narrow_closed_eyes/n_c_patch_'+str(num)+'.jpg',img3);
		    cv2.imshow('visualization', vis3);
		    cv2.waitKey(1);