import numpy as np
import scipy as s

fname = 'gtraw.txt'

with open(fname) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

print content
flag=0
for stf in content:
    print stf[0]
    if stf[0] == '#':
          
          if flag == 1 :
               print label
               np.save(name + '.npy', label)
          label=[]
          name = stf[1:-4]
    else:
       flag=1
       components = stf.split(' ')
       #print components
       frames=components[1]
       nums = frames.split('-')

       value = components[-1]
       if value == 'open':
         res = 1
       elif value == 'closed':
         res=0
       print frames, res
       i = int(nums[0])
       while i<= int(nums[-1]) :
         label.append(res)
         i+=1


