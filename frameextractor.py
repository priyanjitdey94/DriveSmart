#Class responsible for extracting frames from videos

import glob;
import os;
import cv2;

class FrameExtractor:

	def __init__(self):
		self.success=False;
		self.frameCount=0;
		self.videoFilePath="";
		#self.videoFileName=[];

	def resetFrameCount(self):
		self.frameCount=0;

	def setVideoFileName(self,path):
		self.videoFilePath=path;

	def getVideoFileName(self,eyeStatePath):
		if not os.path.exists(eyeStatePath):
			print "Not a proper location of eye_state file.";
			return;

		videos=[];
		for file in open(eyeStatePath):
			file.rstrip('\n');
			videos.append(file);

		return videos;

	def fetchFrame(self,eyeStatePath,eyeState):
		self.videoFileName=self.getVideoFileName(eyeStatePath);
		
		if not os.path.exists(self.videoFilePath+"/"+"Frames_"+eyeState):
			os.makedirs(self.videoFilePath+"/Frames_"+eyeState);

		for file in self.videoFileName:
			try:
				curVideo=cv2.VideoCapture(self.videoFilePath+"/"+file);
			except:
				print "Cannot read "+file;continue;

			try:
				self.success,self.videoFrame=curVideo.read();
			except:
				print "Cannot read frames of "+file;continue;

			self.success=True;
			while self.success:
				self.success,self.videoFrame=curVideo.read();

				str=self.videoFilePath+"/Frames_"+eyeState"/frame"+self.frameCount+".jpg";
				cv2.imwrite(str,self.videoFrame);
				self.frameCount+=1;

			print "Frames of "+file+" fetched successfully.";

			



FE=FrameExtractor();
FE.setVideoFileName("/home/priyanjit/Codes");
FE.fetchFrame();
####
