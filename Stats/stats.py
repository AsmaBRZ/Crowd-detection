import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import norm
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd

def statsByThresh(r0,r1,threshold):
    n0 = len(r0)
    n1 = len(r1)
    trueR0 = 0 
    falseR0 = 0
    trueR1 = 0 
    falseR1 = 0

    for d in r0 : 
        if d <threshold:
            trueR0 = trueR0 + 1
        else :
            falseR0 = falseR0 + 1

    for d in r1 : 
        if d >=threshold:
            trueR1 = trueR1 + 1
        else :
            falseR1 = falseR1 +1

    return trueR0, falseR0, trueR1, falseR1
        
#f0 = open("0_only.txt", "r")
#f1 = open("1_only.txt", "r")

f0 = open("0_only_no_contour.txt", "r")
f1 = open("1_only_no_contour.txt", "r")

r0 = []
r1 = []

while(True):
    line = f0.readline()
    if not line :
        break
    r0.append(round(float(line),2))

while(True):
    line = f1.readline()
    if not line :
        break
    r1.append(round(float(line),2))

f0.close()
f1.close()

r0 = np.array(r0)
r1 = np.array(r1)

"""
hist0, bin_edges0 = np.histogram(r0,bins = np.arange(1.4,1.92,0.01))
hist1, bin_edges1 = np.histogram(r1,bins = np.arange(1.4,1.92,0.01))
plt.hist(r1,bins = np.arange(1.4,1.92,0.01))
plt.hist(r0,bins = np.arange(1.4,1.92,0.01))

plt.title("histogram") 
plt.show()
"""


sns.distplot(r0,bins = np.arange(1.4,1.92,0.01))
sns.distplot(r1,bins = np.arange(1.4,1.92,0.01))

plt.show()


n0 = len(r0)
n1 = len(r1)

thresholds = np.arange(1.7,1.77,0.01)
print(np.arange(1.7,1.77,0.01))
PTR0 = []
PFR0 = []
PTR1 = []
PFR1 = []
l = []
PR=[]
for threshold in thresholds:
    trueR0, falseR0, trueR1, falseR1 = statsByThresh(r0,r1,threshold)
    ptR0 = round(trueR0 / n0,2) * 100
    pfR0 = round(falseR0 / n0,2) * 100

    ptR1 = round(trueR1 / n1,2) * 100
    pfR1 = round(falseR1 / n1,2) * 100

    PTR0.append(ptR0) 
    PFR0.append(pfR0) 
    PTR1.append(ptR1) 
    PFR1.append(pfR1) 

    x10 = [ptR0,pfR0]
    PR.append(x10)

    
    x10 = [ptR1,pfR1]
    PR.append(x10)



    print("Threshold: ",threshold, " ", trueR0," (",ptR0,") ", falseR0, " (",pfR0,") ", trueR1," (",ptR1,") ", falseR1," (",pfR1,") " )

cols = 2
rows = len(thresholds)
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
cp = 0
for i in range(cols):
    for j in range(rows):
        
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid(b=False)
        ax.axis('off')
        ax.pie(PR[cp])
        #ax.set_title("")
        cp = cp + 1
plt.show()
