import caffe
import numpy as np
import sys



blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/prithviraj/eye_lmdb/mean_image.binaryproto' , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
print out
np.save( '/home/prithviraj/eye_lmdb/mean_image.npy' , out )
