#Class responsible recording the video and storing it in a specified location.

class RecordVideo:

	def __init__(self,url,format):
		self.driveLocation=url;
		self.preferedFormat=format;

	def setDriveLocation(self,url):
		self.driveLocation=url;

	def getDriveLocation(self):
		return self.driveLocation;

	def setPreferedFormat(self,format):
		self.preferedFormat=format;

	def getPreferedFormat(self):
		return self.preferedFormat;

	
