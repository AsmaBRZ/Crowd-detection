import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio 
from skimage.filters import meijering
import os
import time
from scipy import stats

threshold_FD = 1.75
parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture ims from camera')
parser.add_argument('--write_video', help='Do you want to write the output video', default=False)
parser.add_argument('--prototxt', help='Path to deploy.prototxt',default='deploy.prototxt', required=False)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel',default='hed_pretrained_bsds.caffemodel', required=False)
parser.add_argument('--width', help='Resize input image to a specific width', default=500, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=500, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
args = parser.parse_args()

def binarizeImage(contour):
  ret, thresh = cv.threshold(contour, 127, 255, 0)
  return thresh

def getFeatures(im):
  imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
  contour = np.zeros(im.shape)

  contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)

  return contour

def smoothImage(im):
  kernel = np.ones((25,25),np.float32)/25
  dst = cv.filter2D(im,-1,kernel)
  
  return dst

             

def fractal_dimension(Z,threshold=0.9):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    Z = (Z < threshold)
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    #print(sizes)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def localize(im):
  blur = smoothImage(im)
  blur = blur.astype(np.uint8)
  contour = np.zeros(im.shape)
  #thresh = binarizeImage(blur)
  print("ttttttttttttttttttttttttttttrrrr",type(blur),blur.shape)
  contours, hierarchy = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1) 
  return contours,countour



class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def makeBlackBorders(im):
  result = im.copy()
  w,h,c = im.shape
  for i in range(w):
    for j in range(h):
      if i == 0 :
        result[i,j] = [0.0,0.0,0.0]
      elif i == w-1 :
        result[i,j] = [0.0,0.0,0.0]

  for i in range(h):
    for j in range(w):
      if i == 0 :
        result[j,i] = [0.0,0.0,0.0]
      elif i == h-1 :
        result[j,i] = [0.0,0.0,0.0]

  return result

start = time.time()
cv.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv.dnn.readNet(args.prototxt, args.caffemodel)

## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'

f = open("FD1.txt", "w")
liste = os.listdir('Images/1/') # dir is your directory path
N = len(liste)
print("Number of elements: ",N)
r = []
for i in range(N):
    im = cv.imread('Images/1/'+str(i)+'.jpg')

    h,w,c = im.shape
    inp = cv.dnn.blobFromImage(im, scalefactor=1.0, size=(args.width, args.height),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (im.shape[1], im.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    #cv.imwrite("out"+str(i)+".png",out)

    #ret, thresh = cv.threshold(out,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    #cv.imwrite("thresh"+str(i)+".png",thresh)

    thresh = cv.adaptiveThreshold(out,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    #cv.imwrite("thresh2.png",thresh)
    thresh = thresh.astype(np.uint8)
    contour = np.zeros(im.shape)
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)
    contour = makeBlackBorders(contour)
    cv.imwrite("Experiments/contour"+str(i)+".png",contour)
    
    #ZZ = cv.imread('contour'+str(i)+'.png',0) 
    

    #ZZ = ZZ.astype(np.uint8)
    #X,Y = localize(ZZ)
    #print("llllllllllllllllllllllllllllllllll",localisation.shape)
    #out=cv.cvtColor(out,cv.COLOR_GRAY2BGR)

    #con=np.concatenate((im,out),axis=1)
    
    #cv.imwrite("localisation"+str(i)+".png",Y)
    I = imageio.imread("Experiments/contour"+str(i)+".png", as_gray="True")/255.0 
    Z = 1.0 - I
    fd = fractal_dimension(Z,threshold=0.5)
    f.write(str(i)+" "+str(fd)+"\n")
    r.append(fd)
    print("Image: ",i," fractal dimension",fd)
    #plt.figure(figsize=(10, 10))
    #plt.imshow(contour)
    #plt.show()
f.close()
end = time.time()

r = np.array(r)
print("Time: ",end-start)
print("Min: ",np.min(r))
print("Max: ",np.max(r))
print("Mean: ",np.mean(r))
print("Median: ",np.median(r))
print("Mode: ",stats.mode(r))

sorted = np.sort(r)
for e in sorted : 
    print(e)

