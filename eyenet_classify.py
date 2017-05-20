import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
import cv2

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

import os
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

caffe.set_mode_cpu()

model_def = '/home/prithviraj/caffe/models/bvlc_googlenet/deploy_eyenet.prototxt'
model_weights = '/home/prithviraj/caffe/models/bvlc_googlenet/eyenet_iter_42000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/home/prithviraj/eye_lmdb/mean_image.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR



net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

def extractPatches(img):
    
    
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
    img2 = img[e_t:e_b,e_l:e_r]

    cv2.imwrite('temp.jpg',img2)
    image = caffe.io.load_image('temp.jpg')
    print image.shape, "SHAPE OF IMAGE"
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    output = net.forward()

    output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

    res = output_prob.argmax()
    print res
    var="close"
    if res ==1:
      var = "open"

    cv2.putText(img,var, (70,70), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255), 8)    
    cv2.imshow("test video", img)
    cv2.waitKey(1)




vids = os.listdir('/home/prithviraj/Downloads/priyanjitdey94-drivesmart-82eb7b6b4664/DatasetBuilder/Video')
opn = np.loadtxt('/home/prithviraj/openface/demos/eyes_open.txt', dtype='str')
cls = np.loadtxt('/home/prithviraj/openface/demos/eyes_test.txt',dtype='str')

i=0

for video in cls[4:]:
   print video
   cap = cv2.VideoCapture('/home/prithviraj/Downloads/priyanjitdey94-drivesmart-82eb7b6b4664/DatasetBuilder/Video/'+video[-1])
   frms = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
   print frms
   f=0
   while(cap.isOpened() and f<frms):
    ret, frame = cap.read()
    extractPatches(frame)
    i+=1
    f+=1
   
    


cap.release()






