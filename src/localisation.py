import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio 
from skimage.filters import meijering
import os
import time 
from scipy import stats

threshold_mask = 0.7 #85%
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
  kernel = np.ones((5,5),np.float32)/25
  dst = cv.filter2D(im,-1,kernel)
  return dst

def smoothPatch(im):
  kernel = np.ones((3,3),np.float32)/9
  dst = cv.filter2D(im,-1,kernel)
  return dst

def localize(im):
  blur = smoothImage(im)
  blur = blur.astype(np.uint8)
  contour = np.zeros(im.shape)
  #thresh = binarizeImage(blur)
  print("ttttttttttttttttttttttttttttrrrr",type(blur),blur.shape)
  contours, hierarchy = cv.findContours(blur, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
  cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1) 
  return contours,countour

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
def countWhitePixels(patch):
  w,h,c=patch.shape
  cp = 0
  for i in range(w):
    for j in range(h):
      if (patch[i,j] == [255.0,255.0,255.0]).all() :
        cp = cp + 1 
  
  return cp
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


cv.dnn_registerLayer('Crop', CropLayer)

# Load the model.
net = cv.dnn.readNet(args.prototxt, args.caffemodel)

## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'

liste = os.listdir('Images/1/') # dir is your directory path
N = len(liste)
print("Number of elements: ",N)
r = []

start = time.time()
for i in range(292,N):
    s0 = time.time()
    im = cv.imread('Images/1/'+str(i)+'.jpg')
    """
    scale_percent = 50 # percent of original size
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    im = cv.resize(im, dim, interpolation = cv.INTER_AREA) 
    """
    im = cv.bilateralFilter(im,9,75,75)
    resize_patch = int(np.max(im.shape)/5)
    print(im.shape,resize_patch)
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
    cv.imwrite("Experiments/edges"+str(i)+".png",out)
    out = binarizeImage(cv.bilateralFilter(out,9,100,100))
    cv.imwrite("Experiments/filtred_edges"+str(i)+".png",out)
    ret, thresh = cv.threshold(out,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imwrite("Experiments/thresh"+str(i)+".png",thresh)
    #thresh = cv.adaptiveThreshold(out,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

    #heatmap_img = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    #cv.imwrite("Experiments/thresh2.png",thresh)
    thresh = thresh.astype(np.uint8)
    contour = np.zeros(im.shape)
    contours,_ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, tuple([255]*im.shape[-1]), 1)
    contour = makeBlackBorders(contour)
    
    
    patches = constructPatches(contour,resize_patch )
    #patches = patches.astype(np.uint8)
    #patches = smoothPatch(patches)

    cv.imwrite("Experiments/contour"+str(i)+".png",contour)

    plt.figure(figsize=(10, 10))
    plt.imshow(patches)
    plt.colorbar()
    plt.savefig("Experiments/patches"+str(i)+".png")
    plt.close()
    #plt.show()
    accepted_value_patch = threshold_mask * np.amax(patches)
    #print("accepted_value_patch",accepted_value_patch)
    mask_patch = patches > accepted_value_patch
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_patch)
    plt.colorbar()
    plt.savefig("Experiments/mask_patch"+str(i)+".png")
    plt.close()
    reconstructed_im = reconstructImage(im,mask_patch,resize_patch)
    cv.imwrite("Experiments/reconstructed_im"+str(i)+".png",reconstructed_im)

    reconstructed_im  = reconstructed_im.astype(np.uint8)
    im_gray = cv.cvtColor(reconstructed_im, cv.COLOR_BGR2GRAY)
    
    contour = im.copy()
    
    contours,_ = cv.findContours(im_gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(contour, contours, -1, (0,0,255), 3)
    cv.imwrite("Experiments/reconstructed_im_contour"+str(i)+".png",contour)
    e0 = time.time()

    print("Image: ",i, " time ",e0 - s0)
    
  
    
end = time.time()
print("Execution time ",end-start)