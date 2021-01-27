import cv2 as cv
import numpy as np
import os
from os.path import dirname, join
import sys

#threshold on the number of white pixels to keep (local density computation step)
threshold_mask = 0.7 #85%

#threshold on the fractal dimension to decide whether a crowd exists or not
threshold_FD = 1.75


#Image binarization
def binarizeImage(contour):
  ret, thresh = cv.threshold(contour, 127, 255, 0)
  return thresh

#image masking
def reconstructImage(im,patch,c):
  w,h,_=im.shape
  reconstructed_im = np.ones((im.shape))
  a = 0
  
  for i in range(0,w,c):
    b = 0
    for j in range(0,h,c):
      lig=0
      col=0

      if i+c <= w :
        lig=i+c 
      else:
        lig=w

      if j+c <= h :
        col=j+c 
      else:
        col=h
      reconstructed_im[i:lig,j:col] *=patch[a,b] * 255.0
      b +=1
    a +=1
  return reconstructed_im

#edge detection
#Source: https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py
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


#delete all the white pixels on the image borders
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

#compute the number of white pixels within a given path (local density)
def countWhitePixels(patch):
  w,h,c=patch.shape
  cp = 0
  for i in range(w):
    for j in range(h):
      if (patch[i,j] == [255.0,255.0,255.0]).all() :
        cp = cp + 1 
  
  return cp

#divide the image into patches
def constructPatches(im,c):
  patches=[]

  w,h,_=im.shape
  for i in range(0,w,c):
    p = []
    for j in range(0,h,c):
      lig=0
      col=0

      if i+c <= w :
        lig=i+c 
      else:
        lig=w

      if j+c <= h :
        col=j+c 
      else:
        col=h

      patch=im[i:lig,j:col]
      #cv.imwrite("Experiments/patch"+str(i)+str(j)+".png",patch)
      #nb_white_pix =  cv2.countNonZero(patch)
      nb_white_pix = countWhitePixels(patch)
      p.append(nb_white_pix)
    patches.append(p)
  
  return np.array(patches)

#compute the fractal dimension of a cotontour map
#Source: https://github.com/ErikRZH/Fractal-Dimension
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
    
#complete crowd localization pipeline
def localize(im):
    global net 
    i  = 1

    #image smoothing
    im = cv.bilateralFilter(im,9,75,75)
    resize_patch = int(np.max(im.shape)/5)

    #edge detection
    inp = cv.dnn.blobFromImage(im, scalefactor=1.0, size=(500,500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (im.shape[1], im.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)
    bf = cv.bilateralFilter(out,9,100,100)

    #image binarization
    out = binarizeImage(bf)
    ret, thresh = cv.threshold(out,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = thresh.astype(np.uint8)

    #contour map construction
    contour = np.zeros(im.shape)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)
    contour = makeBlackBorders(contour)
    
    #divide contour inti patches
    patches = constructPatches(contour,resize_patch )
    accepted_value_patch = threshold_mask * np.amax(patches)
    mask_patch = patches > accepted_value_patch

    #image masking
    reconstructed_im = reconstructImage(im,mask_patch,resize_patch)
    reconstructed_im  = reconstructed_im.astype(np.uint8)
    im_gray = cv.cvtColor(reconstructed_im, cv.COLOR_BGR2GRAY)
    contour = im.copy()
    
    contours,_ = cv.findContours(im_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, (0,0,255), 3)
    cv.imwrite("BB.png",contour)
    
    return contour

#complete crowd classification pipeline
def classify(im):
    i  = 1
    global net

    h,w,c = im.shape
    #edge detection
    inp = cv.dnn.blobFromImage(im, scalefactor=1.0, size=(500,500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (im.shape[1], im.shape[0]))
    out = 255 * out
    out = out.astype(np.uint8)

    #image thresold
    thresh = cv.adaptiveThreshold(out,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    thresh = thresh.astype(np.uint8)

    #contour map construction
    contour = np.zeros(im.shape)
    
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)

    #postprocessing contour map construction => delete white piexels on the image borders
    contour = makeBlackBorders(contour)
    contour = np.array(contour, dtype=np.uint8)
    I = cv.cvtColor(contour, cv.COLOR_BGR2GRAY)/255.0
   
    #comptue fractal dimension
    Z = 1.0 - I
    fd = fractal_dimension(Z,threshold=0.5)
    #print("Image: ",i," fractal dimension",fd)
    
    #decision formulation according to FD
    prediction = 0
    if  fd >= threshold_FD :
        prediction = 1
    return prediction 


#edge detector instantiation
cv.dnn_registerLayer('Crop', CropLayer)

protoPath = join(dirname(__file__), "deploy.prototxt")
modelPath = join(dirname(__file__), 'hed_pretrained_bsds.caffemodel')
# Load the model.
net = cv.dnn.readNet(protoPath, modelPath)

if len(sys.argv) !=2:
    print ("Please specify the image path in arguments")
    sys.exit(1)

im = cv.imread(sys.argv[1])
y = classify(im)
if y ==1:
    print("Class: crowd")
    bb = localize(im)
else:
    print("Class: no crowd")
    cv.imwrite("BB.png",im)

print("Crowd localization image written in local directory")
