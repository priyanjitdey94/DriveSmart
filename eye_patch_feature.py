
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

parser.add_argument('--dlibFaceMean', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "mean.csv"))
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--dlibRoot', type=str,
                    default=os.path.expanduser(
                        "~/src/dlib-18.16/python_examples"),
                    help="dlib directory with the dlib.so Python library.")
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
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


def getRep(imgPath):
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

feat_x = []
feat_y = []
for img in os.listdir('/home/prithviraj/opened_eyes'):
    d = getRep('/home/prithviraj/opened_eyes/'  + img)
    print d.shape
    print "Appending..", img
    feat_x.append(d)
    feat_y.append(1)


for img in os.listdir('/home/prithviraj/closed_eyes'):
    d = getRep('/home/prithviraj/closed_eyes/'  + img)
    print d.shape
    print "Appending..", img
    feat_x.append(d)
    feat_y.append(0)


print "Storing............"
os.mkdir('/home/prithviraj/open_feature')
np.save('/home/prithviraj/open_feature/of.npy', feat_x)
np.save('/home/prithviraj/open_feature/oflabel.npy', feat_y)

feat_x = np.load('/home/prithviraj/open_feature/of.npy')
feat_y= np.load('/home/prithviraj/open_feature/oflabel.npy')
print feat_x.shape, feat_y.shape
print "Training..........."
clf = svm.SVC()
clf.fit(feat_x, feat_y)  
joblib.dump(clf, '/home/prithviraj/open_feature/trained_svm.pkl',compress=1) 

# python /home/prithviraj/openface/demos/compare.py /home/prithviraj/openface/images/examples/*

