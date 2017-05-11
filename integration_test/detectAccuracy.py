
import time

start = time.time()
import argparse
import cv2
import itertools
import os
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
np.set_printoptions(precision=2)

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))

import openface
import openface.helper
from openface.data import iterImgs

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
def getRep(img):
    
    img = cv2.resize(img,(96,96))
    rep = net.forwardImage(img)
    
    return rep

def extractPatches(img,clf):
    
    
    if img is None:
        return
    if args.verbose:
        print("  + Original size: {}".format(img.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(img)
    if bb is None:
      return 
    al = align.align(img,bb)
    
    e_l = al[18][0]
    e_r = al[25][0]
    e_t = al[21][1]
    e_b = al[29][1]
    narrow_e_l = al[36][0]
    narrow_e_r = al[45][0]
    narrow_e_t = min(al[37][1],al[38][1],al[43][1],al[44][1]) 
    narrow_e_b = max(al[41][1], al[40][1],al[46][1],al[47][1])
    #print np.asarray(bb),"<<<<<<"
    #print bb[0], bb[1]
    t = bb.top()
    b = bb.bottom()
    l = bb.left()
    r = bb.right()
    #print l,r,t,b
    img1 = img[t:b,l:r]
    img2 = img[e_t:e_b,e_l:e_r]
    img3 = img[narrow_e_t:narrow_e_b,narrow_e_l:narrow_e_r]
    d = getRep(img2)
    res = clf.predict(d)
    return res


vids = os.listdir('/home/prithviraj/Downloads/priyanjitdey94-drivesmart-82eb7b6b4664/DatasetBuilder/Video')
opn = np.loadtxt('/home/prithviraj/openface/demos/eyes_open.txt', dtype='str')
cls = np.loadtxt('/home/prithviraj/openface/demos/eyes_test.txt',dtype='str')

i=0
clf = joblib.load('/home/prithviraj/open_feature_new/trained_svm.pkl') 
for video in cls[1:]:
   #print video
   tp=0
   total=0
   cap = cv2.VideoCapture('/home/prithviraj/Downloads/priyanjitdey94-drivesmart-82eb7b6b4664/DatasetBuilder/Video/'+video[-1])
   gt_arr = np.load(video[-1][:-4]+'.npy')
   
   #print gt_arr
   frms = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
   #print gt_arr.shape, frms
   #jhf
   #print frms
   f=0
   while(cap.isOpened() and f<frms-1):
    ret, frame = cap.read()

    res = extractPatches(frame,clf)

    #print res
    if res!=None:
       res = res[0]
       print "Valid frame"
       if res == gt_arr[f]:
            print "Its a hit!! :)"
            tp+=1.0
       else :
             print "its a miss :("
       total+=1.0

    
    i+=1
    f+=1
   print "ACCURACY : ", tp/total
   sjvjkgvcfjkfvbxkjcbv
   
    


cap.release()
cv2.destroyAllWindows()


# python /home/prithviraj/openface/demos/compare.py /home/prithviraj/openface/images/examples/*

