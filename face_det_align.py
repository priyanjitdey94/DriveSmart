import time

start = time.time()
import argparse
import cv2
import itertools
import os

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
#net = openface.TorchWrap(args.networkModel, imgDim=args.imgDim, cuda=args.cuda)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(img,num):
    
    
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
    print np.asarray(bb),"<<<<<<"
    #print bb[0], bb[1]
    t = bb.top()
    b = bb.bottom()
    l = bb.left()
    r = bb.right()
    print l,r,t,b
    img1 = img[t:b,l:r]
    img2 = img[e_t:e_b,e_l:e_r]
    img3 = img[narrow_e_t:narrow_e_b,narrow_e_l:narrow_e_r]
    a,b,c = img1.shape
    print a,b,c
    rs_img = cv2.resize(img, (b,a))
    rs_img2 = cv2.resize(img2, (b,a))
    rs_img3 = cv2.resize(img3, (b,a))
    #print "SHAPES:::: ", img1.shape, img2.shape
    vis = np.concatenate((rs_img, img1), axis=1)
    vis2 = np.concatenate((vis, rs_img2), axis=1)
    vis3 = np.concatenate((vis2, rs_img3), axis=1)
    '''
    if bb is None:
        return
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))
    start = time.time()
    alignedFace = align.alignImg("affine", args.imgDim, img, bb)
    if alignedFace is None:
        print "could not align"
        return
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))
    '''
    
    cv2.imwrite('/home/prithviraj/closed_eyes/c_patch_'+str(num)+'.jpg',img2)
    cv2.imwrite('/home/prithviraj/narrow_closed_eyes/n_c_patch_'+str(num)+'.jpg',img3)
    cv2.imshow('visualization', vis3)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
    #cv2.imshow('test',alignedFace)
    
    #cv2.waitKey(0)


vids = os.listdir('/home/prithviraj/Downloads/priyanjitdey94-drivesmart-82eb7b6b4664/DatasetBuilder/Video')
opn = np.loadtxt('/home/prithviraj/openface/demos/eyes_open.txt', dtype='str')
cls = np.loadtxt('/home/prithviraj/openface/demos/eyes_close.txt',dtype='str')

i=0
for video in cls:
   print video
   cap = cv2.VideoCapture('/home/prithviraj/Downloads/priyanjitdey94-drivesmart-82eb7b6b4664/DatasetBuilder/Video/'+video[-1])
   frms = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
   print frms
   f=0
   while(cap.isOpened() and f<frms):
    ret, frame = cap.read()

    getRep(frame,i)
    i+=1
    f+=1
   
    


cap.release()
cv2.destroyAllWindows()


# python /home/prithviraj/openface/demos/compare.py /home/prithviraj/openface/images/examples/*

