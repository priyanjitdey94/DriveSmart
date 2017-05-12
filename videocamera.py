#Class responsible for handling camera opeartions.
import cv2;
import numpy as np;

'''
#startCamera: To start and record video
#stopCamera: Stop camera
#getCameraNum: Return 0 for webcam
#setCameraNum: Set 0 for webcam
#changeInputFormat: change the format in which video is saved.
'''

class CameraClass(object):
	
	def __init__(self,Num,Format):
		self.cameraNum=Num;
		self.videoFormat=Format;

	@staticmethod
	def startCamera(Num):
		print "Starting Camera..... \n";
		cap = cv2.VideoCapture(0)

		# Define the codec and create VideoWriter object
		fourcc = cv2.cv.CV_FOURCC(*'XVID')
		w=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
		h=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
		out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(w),int(h) ));

		while(cap.isOpened()):
		    ret, frame = cap.read()
		    if ret==True:
		        frame = cv2.flip(frame,0)

		        # write the flipped frame
		        out.write(frame)

		        cv2.imshow('frame',frame)
		        if cv2.waitKey(1) & 0xFF == ord('q'):
		            break
		    else:
		        break  

		# Release everything if job is finished
		print "Stopping Camera";
		cap.release()
		out.release()
		cv2.destroyAllWindows()

	@staticmethod
	def stopCamera(cameraNum,vc):
		if vc.isOpened():
			print "Stopping Camera....\n"
			cv2.destroyWindow(cameraNum);
			print "Camera Stopped Successfully.\n";
		else:
			print "Camera not in active mode.";

	def getCameraNum(self):
		return self.cameraNum;

	def setCameraNum(self,num):
		self.cameraNum=num;

	def changeInputFormat(self,newFormat):
		if newFormat not in ["mp4","mkv","3gp"]:
			print "Invalid video format, cancelling request.\n";
		else:
			self.videoFormat=newFormat;
			print "Video format changed successfully.\n";


#cc=CameraClass(0,"mp4");
#cc.startCamera();
