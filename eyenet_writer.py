import os
import numpy as np
output = open('/home/prithviraj/output_eyenet.txt','w')

fil = os.listdir('/home/prithviraj/closed_eyes')

for stf in fil :
    address = '/home/prithviraj/closed_eyes/' + stf
    label = '0'
    output.write(address + ' ' + label + '\n')




fil = os.listdir('/home/prithviraj/opened_eyes')

for stf in fil :
    address = '/home/prithviraj/opened_eyes/' + stf
    label = '1'
    output.write(address + ' ' + label + '\n')
    	
    
