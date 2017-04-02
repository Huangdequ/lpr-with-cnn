#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from math import sqrt,cos
import time
import sys
import tools
import scipy.io as sio
###################################
#######  Local the license  #######
###################################
time1=time.time()
num=input('Please enter the number:')
file_path='/home/hdq/chepai/%s.jpg'%num
img=cv2.imread(file_path)
size=img.shape
scale=sqrt((float(size[0])*float(size[1]))/1190400)
#print ('the size and scale:%s,%s'%(size,round(scale)))
if size[0]*size[1]>1190400 and scale>1:
	img=cv2.resize(img,(int(size[1]/scale),int(size[0]/scale)),interpolation=cv2.INTER_CUBIC)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gasus=cv2.GaussianBlur(gray,(5,5),0)
#blur=cv2.medianBlur(gray,5)
cv2.imshow('img',img)
hrc=tools.hsvfilter(img)
cv2.imshow('hrc',hrc)
kernel = np.ones((15,15),np.uint8)
tophat= cv2.morphologyEx(gasus,cv2.MORPH_TOPHAT,kernel)
tophat=cv2.medianBlur(tophat,5)
edge=cv2.Sobel(tophat,-1,1,0)
#edge=cv2.blur(edge,(3,3))
cv2.imshow('edges',edge)
ret,binary=cv2.threshold(edge,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(30,7))
kernel2=cv2.getStructuringElement(cv2.MORPH_RECT,(4,1))
closing=cv2.dilate(binary,kernel2,iterations=4)
closing=cv2.morphologyEx(closing,cv2.MORPH_CLOSE,kernel1,iterations=1)
closing=cv2.erode(closing,kernel2,iterations=4)


dst=cv2.bitwise_and(closing,hrc)

dst=cv2.dilate(dst,kernel2,iterations=1)
dst=cv2.erode(dst,kernel2,iterations=1)
kernel3=cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
dst=cv2.dilate(dst,kernel3,iterations=1)

#print time3-time1
cv2.imshow('erode',closing)
cv2.imshow('dst',dst)
print dst.shape
#Choose by the width and heigh
contours,heirs = cv2.findContours(dst.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rectangle=[]
rotation=0
smax=0
for tour in contours:
	rc=cv2.boundingRect(tour)
	msc=cv2.minAreaRect(tour)
	
	if len(contours) == 1:
		rectangle.append(rc)
		if msc[1][1]>msc[1][0]:
			rotation=msc[2]+90
		else:
			rotation=msc[2]
		print msc
		print rc
	elif abs((rc[2]/rc[3])-22/7)<2.4 and rc[2]*rc[3]>max(600,dst.shape[0]*dst.shape[1]/600):
		print msc
		print rc
		if abs(max(msc[1])/min(msc[1])-4.0)<1.8:
			temp = img[rc[1]:(rc[1]+rc[3]),rc[0]:(rc[0]+rc[2])]
			percent=dst[rc[1]:(rc[1]+rc[3]),rc[0]:(rc[0]+rc[2])].mean()/255
			tem = cv2.cvtColor(temp,cv2.COLOR_BGR2HSV)
			h=tem[:,:,0].mean()
			s=tem[:,:,1].mean()
			print h,s
			if h<130 and h>90 and s>70 :
				if len(rectangle)!=0:
					
					if (s+sqrt(rc[2]*rc[3]))*percent>smax:
						del rectangle[0]	
						rectangle.append(rc)
						smax=s+sqrt(rc[2]*rc[3])*percent
						if msc[1][1]>msc[1][0]:
							rotation=msc[2]+90
						else:
							rotation=msc[2]
				else:
					rectangle.append(rc)
					smax=s+sqrt(rc[2]*rc[3])*percent
					if msc[1][1]>msc[1][0]:
						rotation=msc[2]+90
					else:
						rotation=msc[2]
#Choose by the color
print('The rectangle has:%d'%len(rectangle))

if len(rectangle)== 1:
	ar = rectangle[0]
	yy=ar[1]-5
	if yy<0:
		yy=0
	xx=ar[0]-5
	if xx<0:
		xx=0
	temp = img[yy:(ar[1]+ar[3]+2),xx:ar[0]+ar[2]+5]
	cv2.imwrite('img.jpg',temp)
else:
	print 'narrow'
	gasus=cv2.resize(gasus,(int(size[1]/2),int(size[0]/2)),interpolation=cv2.INTER_CUBIC)
	img=cv2.resize(img,(int(size[1]/2),int(size[0]/2)),interpolation=cv2.INTER_CUBIC)
	hrc=tools.hsvfilter(img)
	cv2.imshow('hrc',hrc)
	kernel = np.ones((15,15),np.uint8)
	tophat= cv2.morphologyEx(gasus,cv2.MORPH_TOPHAT,kernel)
	tophat=cv2.medianBlur(tophat,5)
	edge=cv2.Sobel(tophat,-1,1,0)
	#edge=cv2.blur(edge,(3,3))
	cv2.imshow('edges',edge)
	ret,binary=cv2.threshold(edge,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(30,7))
	kernel2=cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
	closing=cv2.dilate(binary,kernel2,iterations=4)
	closing=cv2.morphologyEx(closing,cv2.MORPH_CLOSE,kernel1,iterations=1)
	closing=cv2.erode(closing,kernel2,iterations=4)
	dst=cv2.bitwise_and(closing,hrc)
	dst=cv2.dilate(dst,kernel2,iterations=1)
	dst=cv2.erode(dst,kernel2,iterations=1)
	kernel3=cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
	dst=cv2.dilate(dst,kernel3,iterations=1)
	cv2.imshow('erode',closing)
	cv2.imshow('dst',dst)
	print dst.shape
	#Choose by the width and heigh
	contours,heirs = cv2.findContours(dst.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	rectangle=[]
	rotation=0
	smax=0
	for tour in contours:
		rc=cv2.boundingRect(tour)
		msc=cv2.minAreaRect(tour)
		
		if len(contours) == 1:
			rectangle.append(rc)
			if msc[1][1]>msc[1][0]:
				rotation=msc[2]+90
			else:
				rotation=msc[2]
			print msc
			print rc
		elif rc[2]>rc[3] and abs((rc[2]/rc[3])-22/7)<2.4 and rc[2]*rc[3]>max(600,dst.shape[0]*dst.shape[1]/600):
			print msc
			print rc
			if abs(max(msc[1])/min(msc[1])-3.8)<1.8:
				temp = img[rc[1]:(rc[1]+rc[3]),rc[0]:(rc[0]+rc[2])]
				tem = cv2.cvtColor(temp,cv2.COLOR_BGR2HSV)
				h=tem[:,:,0].mean()
				s=tem[:,:,1].mean()
				print h,s
				if h<130 and h>80 and s>70 :
					if len(rectangle)!=0:
						
						if (s+sqrt(rc[2]*rc[3]))>smax:
							del rectangle[0]	
							rectangle.append(rc)
							smax=s+sqrt(rc[2]*rc[3])
							if msc[1][1]>msc[1][0]:
								rotation=msc[2]+90
							else:
								rotation=msc[2]
					else:
						rectangle.append(rc)
						smax=s+sqrt(rc[2]*rc[3])
						if msc[1][1]>msc[1][0]:
							rotation=msc[2]+90
						else:
							rotation=msc[2]		
	if len(rectangle)==0:
		print('error!Can not locate the words!')
	else:
		ar = rectangle[0]
		yy=ar[1]-5
		if yy<0:
			yy=0
		xx=ar[0]-5
		if xx<0:
			xx=0
		temp = img[yy:(ar[1]+ar[3]+2),xx:ar[0]+ar[2]]
		cv2.imwrite('img.jpg',temp)
		
###################################
#######   Cut the license   #######
###################################

img=cv2.imread('img.jpg')
cv2.imshow('iimg',img)
cv2.waitKey(0)

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(100,100))
tw=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel1)
bw=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel2)
temp=cv2.add(img,tw)
result=cv2.subtract(temp,bw)
ret,bw=cv2.threshold(result,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('final',result)

bw=tools.rotate_about_center(bw,rotation,flag=False)
ret,bw=cv2.threshold(bw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#bw=tools.PcaRotate(bw)
cv2.imshow('rotate',bw)

smallflag=False
if abs(rotation)<5:
	bwheigh_10=0
	if abs(rotation)<=1 and bw.shape[0]*bw.shape[1]<5000:
		smallflag=True
		rotation=0
else:
	bwheigh_10=bw.shape[0]/20

nepoch=0
wordsnum=0
bwscale=0

while nepoch<8 and wordsnum<6:
	bw=bw[0:bw.shape[0]-nepoch,:] 
	bw=tools.locatethewords(bw,bwheigh_10,smallflag)
	bw=tools.cuttheimg(bw,flag=False)

	if nepoch==0:
		bwscale=8.8*cos(np.deg2rad(rotation))*cos(np.deg2rad(rotation))/(bw.shape[1]/float(bw.shape[0]))
    	print 'the bwscale is:',bwscale
	if bw.shape[0]*bw.shape[1]==0:
		print'Error,could not find th words!'
		sys.exit(0)
	words,isone,wordsnum=tools.cutthewords(bw,rotation,bwscale)
	nepoch+=1
print'nepoch',nepoch
cv2.imshow('qiege',bw)


num=len(words)
print'The word has:%s'%num
wordsort=[words[i][0] for i in xrange(num)]
arrayisone=np.array(isone)
isone=arrayisone[np.array(np.argsort(wordsort))].tolist()
print isone

if num>=1:

	words=sorted(words,key=lambda x:x[0])
	print words	
	while num>7:
		if (words[1][0]-words[0][0])<(words[2][2]/2) or words[0][3]<words[3][3]/2 or (words[1][3]+words[2][3])/2-words[0][3]>words[2][3]/6:
			del words[0]
			del isone[0]
			num-=1
		else:
			del words[num-1]
			del isone[num-1]
			num-=1
	if num==7 :
		if words[6][0]-words[5][0]-words[5][2]<words[6][2]-5 and isone[6]==1 :
			del words[num-1]
			del isone[num-1]
			tem=[0,0,0,0]
			tem[0]=int(round(words[0][0]-words[0][2]))
			while sum(bw[:,tem[0]])==0:
				tem[0]+=1
			while sum(bw[:,tem[0]])>255*2 and tem[0]>0:
				tem[0]-=1
			if tem[0]<0:
				tem[0]=0
			print '\ntem0:',tem[0]
			tem[1]=words[0][1]
			tem[2]=int(round(words[1][2]*1.4))
			print 'words[0][0]:',words[0][0]
			print 'sum:',(tem[2]+tem[0])
			if tem[2]+tem[0]>=words[0][0]:
				tem[2]=words[0][0]-tem[0]-2
			tem[3]=words[0][3]
			tem=tuple(tem)
			words.insert(0,tem)
			isone.insert(0,0)
		else:
			tem=[0,0,0,0]
			tem[0]=int(round(words[1][0]-words[1][2]))
			while sum(bw[:,tem[0]])>255*2 and tem[0]>0:
				tem[0]-=1
			if tem[0]<0:
				tem[0]=0			
			tem[1]=words[1][1]
			tem[2]=int(round(words[1][2]*1.4))
			if (tem[2]+tem[0])>=words[1][0]:
				tem[2]=words[1][0]-tem[0]-2
			tem[3]=words[1][3]
			tem=tuple(tem)
			del words[0]
			words.insert(0,tem)
	elif num==6:
		if words[0][0]<words[0][2]*2/3:
			tem=[0,0,0,0]
			tem[0]=int(round(words[num-1][0]+words[num-1][2]*1.1))
			while sum(bw[:,tem[0]])==0 and tem[0]<bw.shape[1]-2:
				tem[0]+=1
			tem[1]=words[num-1][1]
			tem[2]=tem[0]+1
			while sum(bw[:,tem[2]])>0 and tem[2]<bw.shape[1]-2:
					tem[2]+=1
			tem[3]=words[num-1][3]
			tem=tuple(tem)
			words.append(tem)
			if tem[2]/tem[3]>4:
				isone.append(1)
			else:
				isone.append(0)
			num+=1

		else:
			tem=[0,0,0,0]
			tem[0]=int(round(words[0][0]-words[0][2]))
			while sum(bw[:,tem[0]])>255*2 and tem[0]>0:
				tem[0]-=1
			if tem[0]<0:
				tem[0]=0
			tem[1]=words[0][1]
			tem[2]=int(round(words[1][2]*1.4))
			if (tem[2]+tem[0])>=words[0][0]:
				tem[2]=words[0][0]-tem[0]-2
			tem[3]=words[0][3]
			tem=tuple(tem)
			words.insert(0,tem)
			isone.insert(0,0)
			num+=1

else:
	print('cannot find the words!!')
print isone
print bw.shape
time2=time.time()
#print time2-time1
print words
cv2.waitKey(0)
cv2.destroyAllWindows()

fig=plt.figure()
for i in xrange(num):
	plt.subplot(2,num,i+1)
	ar=words[i]
	temp=bw[int(ar[1]):int(ar[1]+ar[3]+1),int(ar[0]):int(ar[0]+ar[2]+1)]

	temp=tools.cuttheimg(temp,flag=False)	
	plt.imshow(temp,cmap='gray')
	plt.axis("off")
	temp=cv2.resize(temp,(14,28),interpolation=cv2.INTER_AREA)

	cv2.imwrite('%s.jpg'%(i+1),temp)

if num<7:
	print 'Your should enter 7 words!'
	sys.exit(0)

###################################
##### recongnite the license ######
###################################
input_vector=[]
for i in xrange(7):
	plt.subplot(2,num,i+8)
	file_path='%s.jpg'%(i+1)
	img=cv2.imread(file_path)
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,bw=cv2.threshold(img,0,255,200)
	high,wide=bw.shape	
	czeros=np.zeros((28,7),np.uint8)
	final=np.column_stack((czeros,bw,czeros))
	cv2.imshow('bw',bw)
	vector=final.reshape((784,))
	input_vector.append(vector)
	plt.imshow(final,cmap ='gray')
	plt.axis("off")
plt.show()
inputv=np.array(input_vector)
inputv=np.transpose(inputv)
test_x=inputv
cnn_hanzi=sio.loadmat('cnn_hanzi.mat')
cnn=cnn_hanzi['cnn']

word=test_x[:,0].reshape((28,28))
cnn = tools.cnnff2(cnn,word)
h1=np.argmax(cnn['o'][0][0])
h1=tools.get_hanzi(h1+1)
cnn_26zimu=sio.loadmat('cnn_zimu.mat')
cnn=cnn_26zimu['cnn']
word=test_x[:,1].reshape((28,28))
cnn = tools.cnnff2(cnn,word)
h2=np.argmax(cnn['o'][0][0])
h2=tools.get_26zimu(h2+1)
cnn_zimu_shuzi=sio.loadmat('cnn_zimushuzi.mat')
cnn=cnn_zimu_shuzi['cnn']
h3_7=[0,0,0,0,0]

print '\n  The number of licence plate is:'
print h1,h2,
for i in range(2,7):
	if isone[i]==1:
		h3_7[i-2]='1'
		print h3_7[i-2],
	else:
		word=test_x[:,i].reshape((28,28))
		cnn = tools.cnnff2(cnn,word)
		h3_7[i-2]=tools.get_zimu_shuzi(np.argmax(cnn['o'][0][0])+1)
		print h3_7[i-2],
time3=time.time()
