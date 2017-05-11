#Frame class

class Frame(object):

	def __init__(self,Num):
		self.frameNumber=Num;
		self.frameFormat="jpg";

	def getFrameNum(self):
		return self.frameNumber;

	def setFrameNum(self,Num):
		return self.frameNumber;
