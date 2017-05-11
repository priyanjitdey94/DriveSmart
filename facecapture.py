

import cv2;
import numpy as np;

from videoCamera import CameraClass;

class FaceCapture(CameraClass):

	def isCameraWorking(self):
		cv2.namedWindow("preview")
		vc = cv2.VideoCapture(0)

		if vc.isOpened(): # try to get the first frame
			rval, frame = vc.read()
		else:
			rval = False

		cv2.destroyWindow("preview")

		return rval;
