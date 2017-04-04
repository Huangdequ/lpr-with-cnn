#! usr/bin/env python
# coding=utf-8 
import cv2
import numpy as np
import math
import scipy.io as sio
from numpy.matlib import repmat 
import copy
import sys
def zeroMean(dataMat):        
    meanVal=np.mean(dataMat,axis=0)    
    newData=dataMat-meanVal  
    return newData,meanVal 
def percentage2n(eigvals):
	sortarray=np.sort(eigvals)
	sortarray=sortarray[-1::-1]
	arraysum=sum(sortarray)
	tmpsum=0
	num=0
	for i in sortarray:
		tmpsum+=i
		num+=1
		if tmpsum>=arraysum*0.9:
			return num

def pca(dataMat):
	newData,meanVal=zeroMean(dataMat) 
	covMat=np.cov(newData,rowvar=0)
	eigVals,eigVects=np.linalg.eig(np.mat(covMat))
	n=percentage2n(eigVals)
	eigValIndice=np.argsort(eigVals)
	n_eigValIndice=eigValIndice[:-(n+1):-1] 
	n_eigVect=eigVects[:,n_eigValIndice]       
	return covMat,n_eigVect
def rotate_about_center(src,rangle,flag=True,scale=1.0):
    w=src.shape[1]
    h=src.shape[0]
    if flag== True:
        rangle=math.pi/2-rangle
        angle=np.rad2deg(rangle)
        if angle>90:
    	   angle=180+angle
    else:
        angle=rangle
        rangle=np.deg2rad(angle)
    nw=(abs(math.sin(rangle)*h)+abs(math.cos(rangle)*w))*scale
    nh=(abs(math.cos(rangle)*h)+abs(math.sin(rangle)*w))*scale
    rot_mat=cv2.getRotationMatrix2D((nw*0.5,nh*0.5),angle,scale)
    rot_move=np.dot(rot_mat,np.array([(nw-w)*0.5,(nh-h)*0.5,0]))
    rot_mat[0,2]+=rot_move[0]
    rot_mat[1,2]+=rot_move[1]
    return cv2.warpAffine(src,rot_mat,(int(math.ceil(nw)),int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
def PcaRotate(img):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask[:] = 0 
    cv2.floodFill(img, mask,(w-1,1),0)
    cv2.floodFill(img, mask,(1,h-1),0)
    cv2.floodFill(img, mask,(w-1,h-1),0)
    cv2.floodFill(img, mask,(1,1),0)
    temp=np.where(img==255)
    location=np.array(temp)
    location=location.T
    conmat,base=pca(location)
    theta=math.atan(base[1,0]/base[0,0])
    src=rotate_about_center(img,theta)
    ret,src=cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return src
def hsvfilter(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    heigh=hsv.shape[0]
    width=hsv.shape[1]   
    lut1=np.zeros(256)
    lut2=np.zeros(256)
    lut3=np.zeros(256)
    for i in range(0,255):
        if i<130 and i>97:
            lut1[i]=255
        if i>70:
            lut2[i]=255
        if i>55:
            lut3[i]=255
    im1=cv2.LUT(hsv[:,:,0],lut1)
    im2=cv2.LUT(hsv[:,:,1],lut2)
    im3=cv2.LUT(hsv[:,:,2],lut3)
    im=cv2.bitwise_and(im1,im2)
    ims=cv2.bitwise_and(im,im3)
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(4,1))
    ims=cv2.dilate(ims,kernel,iterations=1)
    ims=cv2.morphologyEx(ims,cv2.MORPH_CLOSE,kernel,iterations=2)
    ims=np.uint8(ims)
    return ims

def locatethewords(img,shift=0,smallflag=False):
    if img.shape[0]*img.shape[1]>12000:
        kernel=(5,5)
        thb=False
    else:
        kernel=(3,3)
        thb=True
    bw=cv2.GaussianBlur(img,kernel,0)
    ret,bw=cv2.threshold(bw,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    width=bw.shape[1]
    heigh=bw.shape[0]
    #qiege
    yy1=0
    yy2=heigh-1
    if smallflag:
        p=12-2
    else:
        p=12
    center=heigh/2

    for y in xrange(center):
        npixels=0
        lpixels=0
        jump=0
        for x in xrange(width):
            if bw[heigh/2-y,x]!=jump:
                jump=bw[center-y,x]
                npixels+=1
        
        if npixels<p or (center-y)==0:
            for x in xrange(width):
                if bw[heigh/2-y-1,x]!=jump:
                    jump=bw[center-y,x]
                    lpixels+=1
            if lpixels>7 and thb:
                print'thbup'
                yy1=heigh/2-y-shift
                if yy1<0:
                    yy1=0
            else:
                yy1=heigh/2-y
            break

    for y in xrange(heigh-center):
        npixels=0
        lpixels=0
        jump=0
        for x in xrange(width):
            if bw[center+y,x]!=jump:
                jump=bw[center+y,x]
                npixels+=1  

        if npixels<p or y==heigh/2:
            for x in xrange(width):
                if bw[heigh/2+y-1,x]!=jump:
                    jump=bw[center+y,x]
                    lpixels+=1
            if lpixels>7 and thb:
                print'thbdown'
                yy2=heigh/2+y+shift
                if yy2>heigh:
                    yy2=heigh
            else:
                yy2=heigh/2+y
            break

    cbw=img[yy1:yy2,:]
    return cbw
def cuttheimg(img,flag=True,x=True,y=True):
    ret,img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if img==None:
        print 'error!'
        sys.exit(0)
    sp=img.shape
    if sp[1]/sp[0]>6:
        pass 
    if flag:
        for i in xrange(sp[1]):
            if sum(img[:,i])<=510:
                img[:,i]=0
    xx1=0
    xx2=sp[1]-1
    yy1=0
    yy2=sp[0]-1
    if y:
        for i in xrange(sp[0]):
            if sum(img[i,:])>0:
                yy1=i
                break
        for i in xrange(sp[0]):
            if sum(img[sp[0]-i-1,:])>0:
                yy2=sp[0]-i-1
                break
    if x:
        for j in xrange(sp[1]):
            if sum(img[:,j])>0 :
                xx1=j
                break
        for j in xrange(sp[1]):
            if sum(img[:,sp[1]-1-j])>0 :
                xx2=sp[1]-1-j
                break
    if yy1-1<0:
        yy1=0
    else:
        yy1-=1
    if xx1-1<0:
        xx1=0
    else:
        xx1-=1    
    cut=img[yy1:yy2+1,xx1:xx2+1]
    return cut

def cutthewords(bw,rotation,bwscale):
        wordsnum=0
        print 'the bw.shape of license is:',bw.shape
       
        contours,heirs = cv2.findContours(bw.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for inner in contours:
            rc=cv2.boundingRect(inner)
            if abs(rc[3]/rc[2]-3.2)<1 and rc[3]*rc[2]>(bw.shape[1]*bw.shape[0]*1/3):
                bw=bw[int(rc[1]):int(rc[1]+rc[3]),int(rc[0]):int(rc[0]+rc[2])]
                
        words=[]
        isone=[]
        contours,heirs = cv2.findContours(bw.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for tour in contours:
            rc=cv2.boundingRect(tour)
            
            if rc[3]>rc[2] and abs((float(rc[3])/rc[2])-bwscale)<=1 and bw[rc[1]:rc[3]+rc[1],rc[0]:rc[0]+rc[2]].mean()<220 and rc[3]*rc[2]>bw.shape[0]*2:
                words.append(rc)
                isone.append(0)
                wordsnum+=1
            else:
                msc=cv2.minAreaRect(tour)

                if rc[3]>rc[2] and msc[1][1]*msc[1][0]!=0 and abs(max(msc[1])/min(msc[1])-4.5*bwscale)<=4 and \
                    rc[0] >=(bw.shape[1]/3) and rc[2]*rc[3]>(bw.shape[0]*1.2):
                    print msc
                    rsc=[]
                    rsc.append(rc[0]-rc[2]/2)
                    rsc.append(rc[1])
                    rsc.append(rc[2]*2)
                    rsc.append(rc[3])
                    rsc=tuple(rsc)
                    words.append(rsc)
                    isone.append(1)
                    wordsnum+=1

        return words,isone,wordsnum
def get_hanzi(h1):
    if h1== 1: h1='藏'
    elif h1== 2: h1='川'
    elif h1== 3: h1='鄂'
    elif h1== 4: h1='甘'
    elif h1== 5: h1='赣'
    elif h1== 6: h1='贵'
    elif h1== 7: h1='桂'
    elif h1== 8: h1='黑'
    elif h1== 9: h1='沪'
    elif h1== 10: h1='吉'
    elif h1== 11: h1='冀'
    elif h1== 12: h1='津'
    elif h1== 13: h1='晋'
    elif h1== 14: h1='京'
    elif h1== 15: h1='辽'
    elif h1== 16: h1='鲁'
    elif h1== 17: h1='蒙'
    elif h1== 18: h1='闽'
    elif h1== 19: h1='宁'
    elif h1== 20: h1='青'
    elif h1== 21: h1='琼'
    elif h1== 22: h1='陜'
    elif h1== 23: h1='苏'
    elif h1== 24: h1='皖'
    elif h1== 25: h1='湘'
    elif h1== 26: h1='新'
    elif h1== 27: h1='渝'
    elif h1== 28: h1='豫'
    elif h1== 29: h1='粤'
    elif h1== 30: h1='云'
    elif h1== 31: h1='浙'
    return h1
def get_26zimu(h2):
    h2=chr(h2+64)
    return h2
 
def get_zimu_shuzi(h):
    if h<=8:           # A ~ H
        h=chr(h+64)
    elif 9<=h<=13:     # J ~ N
        h=chr(h+65)
    elif 14<=h<=24:    # P ~ Z
        h=chr(h+66)
    elif h==25:
        h=0
    elif 26<=h<=33:     # 数字
        h=h-24
    return h

def sigmoid(x):
    hx=x.shape[0]
    lx=x.shape[1]
    res=[1/(1+math.exp(-x[i][j]))  if x[i][j]>0 else 1-1/(1+math.exp(x[i][j])) for i in range(hx) for j in range(lx)]
    y=np.array(res).reshape((hx,lx))
    return y 


def conv(x,y):
    bx=x.shape[0]
    by=y.shape[1]
    bz=bx-by+1
    zs=[sum([np.convolve(x[i+k,j:j+by],y[by-1-i,],'vaild')for i in xrange(by)])for k in xrange(bz)for j in xrange(bz)]
    z=np.array(zs).reshape((bz,bz))
    return z

def sub(x,y):
 #x=np.array(range(25)).reshape(5,5)
 #y=np.array(range(9)).reshape(3,3)
    bx=x.shape[0]
    by=y.shape[0]
    bz=bx/by
    zs=[sum([np.convolve(x[i+k,j:j+by],y[by-1-i,],'valid')for i in range(by)])for k in range(bx)[0:bx:by]for j in range(bx)[0:bx:by]]
 #z=np.zeros([3,3])
    z=np.array(zs).reshape((bz,bz))
    return z

def cnnff2(cnn,word):
    layers=cnn['layers'][0,0]
    conv1=layers[1,0]
    sub1=layers[2,0]
    conv2=layers[3,0]
    subl2=layers[4,0]
    n=layers.size
    layers[0,0]['a'][0,0][0,0]=word
    inputmaps=1
    for i in range(1,n):
        layer2=layers[i][0]
        layer1=layers[i-1][0]
        if layer2[0][0][0][0]=='c':
            for j in xrange(layer2['outputmaps'][0,0][0,0]):
                z=np.zeros(layer1['a'][0,0][0,0].shape[0]-layer2['kernelsize'][0,0][0,0]+1,layer1['a'][0,0][0,0].shape[0]-layer2['kernelsize'][0,0][0,0]+1)
                for k in xrange(inputmaps):
                    dst=copy.copy(layer1['a'][0,0][0,k-1])
                    w=copy.copy(layer2['k'][0,0][0,k-1][0,j])
                    z=z+conv(dst,w)                     
                layer2['a'][0,0][0,j]=sigmoid(z+layer2['b'][0,0][0,j])
            inputmaps=layer2['outputmaps'][0,0][0,0]
        elif layer2[0][0][0][0]=='s':
            for j in xrange(inputmaps):
                dst=copy.copy(layer1['a'][0,0][0,j])
                w=np.ones((layer2['scale'][0,0][0,0],layer2['scale'][0,0][0,0]))/pow(layer2['scale'][0,0][0,0],2)
                layer2['a'][0,0][0,j]=sub(dst,w)
    fv=cnn['fv'][0,0]
    for s in range(layers[n-1,0]['a'][0,0].size):
        if s ==0:
            sa=layers[n-1,0]['a'][0,0][0,s].shape
            tem=np.transpose(layers[n-1,0]['a'][0,0][0,s])
            fv=tem.reshape((sa[0]*sa[1],1))
        else:
            sa=layers[n-1,0]['a'][0,0][0,s].shape
            fv=np.row_stack((fv,np.transpose(layers[n-1,0]['a'][0,0][0,s]).reshape((sa[0]*sa[1],1))))
    W=np.dot(cnn['ffW'][0,0],fv)
    b=repmat(cnn['ffb'][0,0],1,fv.shape[1])
    cnn['o'][0,0]=sigmoid(W+b)
    return cnn  