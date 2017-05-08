#Class responsible for handling camera opeartions.

import cv2

class CameraClass:
	
	def __init__(self,Num,Format):
		self.cameraNum=Num;
		self.videoFormat=Format;

	def startCamera(self):
		print "Starting Camera..... \n";
		cv2.namedWindow("DriveSmart");
		vc = cv2.VideoCapture(self.cameraNum);

		if vc.isOpened():
			rval,frame=vc.read();
			print "Camera started recording.\n";
		else:
			rval = false;
			print "Camera not started. Failure.";

		while rval:
			cv2.imshow("DriveSmart",frame); #show video
			rval,frame = vc.read();
			key = cv2.waitKey(27);
			if key==27:
				break;

		vc.release;
		cv2.destroyWindow("DriveSmart");
		print "Camera stopped successfully";
		#self.stopCamera("DriveSmart",vc);

	def stopCamera(self,cameraNum,vc):
		if vc.isOpened():
			print "Stopping Camera....\n"
			cv2.destroyWindow(cameraNum);
			print "Camera Stopped Successfully.\n";
		else:
			print "Camera not in active mode.";

	def getCameraNum(self):
		return self.cameraNum;

	def changeInputFormat(self,newFormat):
		if newFormat not in ["mp4","mkv","3gp"]:
			print "Invalid video format, cancelling request.\n";
		else:
			self.videoFormat=newFormat;
			print "Video format changed successfully.\n";


#cc = CameraClass(0,"mp4");
#cc.startCamera();

'''
from recordvideo import RecordVideo;
RC=RecordVideo("url","mp4");
RC.test();
print RC.getDriveLocation();
RC.setDriveLocation("llll");
print RC.getDriveLocation();
'''
