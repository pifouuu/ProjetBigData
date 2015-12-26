import os
import numpy as np
#assumes labelled data ra stored into a positive and negative folder
#returns two lists one with the text per file and another with the corresponding class 
def loadLabeled(path, nb):

	rootdirPOS =path+'/pos'
	rootdirNEG =path+'/neg'
	data=[]
	Class=[]
	count=0
	nbPosFiles=0
	for subdir, dirs, files in os.walk(rootdirPOS):
		for posFileIdx in range(nb):
			with open(rootdirPOS+"/"+files[posFileIdx], 'r') as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
	tmpc1=np.ones(len(data))
	for subdir, dirs, files in os.walk(rootdirNEG):
		for negFileIdx in range(nb):
			with open(rootdirNEG+"/"+files[negFileIdx], 'r') as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
	tmpc0=np.zeros(len(data)-len(tmpc1))
	Class=np.concatenate((tmpc1,tmpc0),axis=0)
	return data,Class
#loads unlabelled data	
#returns two lists
#one with the data per file and another with the respective filenames (without the file extension)
def loadUknown(path,nb):
	rootdir=path
	data=[]
	names=[]
	for subdir, dirs, files in os.walk(rootdir):
		for fileIdx in range(nb):
			with open(rootdir+"/"+files[fileIdx], 'r') as content_file:
				content = content_file.read() #assume that there are NO "new line characters"
				data.append(content)
				names.append(files[fileIdx].split(".")[0])
	return data,names