import numpy


import numpy as np
import numpy.linalg as linalg
import sys
import cv2
import math
import scipy.stats
from PIL import Image


def common(img1 ,img2):
    h, w = img1.shape[:2]
    print h
    width = 0
    mini = 1000
    for i in range(1, w-1, 15):
        diff = np.abs(np.sum(img1[:][w-1-i:w-1] - img2[:][0:i]))/(i*h)
        if diff < mini:
            mini = diff
            width = i
    return width


def init_warp(w, p):
    w[0][0] = 1 + p[0]
    w[0][1] = p[2]
    w[0][2] = p[4]
    w[1][0] = p[1]
    w[1][1] = 1 + p[3]
    w[1][2] = p[5]


def stitching(img1, img2, u, v, width):

    offset = 10
    h = max(img1.shape[0], img2.shape[0]) + np.abs(v)
    w = img1.shape[1] + img2.shape[1] - width + offset
    print h, w
    stitched_img = np.zeros((h, w), dtype=np.uint8)
    if v < 0:
        im1h, im1w = img1.shape[:2]
        # print img1.shape
        v = np.abs(v)
        stitched_img[v:v+im1h, 0:im1w] = img1[0:im1h, 0:im1w]
        sw = img1.shape[1]-width-u
        # stitched_img[0 : im1h, sw : sw + im1w] = (stitched_img[0 : im1h, sw : sw + im1w] + img2)/2
        stitched_img[0 : im1h, sw : sw + im1w] = img2[0:im1h, 0:im1w]
        stitched_img[im1h/2:im1h, sw : sw + 70] = (img2[im1h/2:im1h, 0:70] + img1[im1h/2:im1h, sw:sw+70])/2
        cv2.GaussianBlur(stitched_img[0 : im1h, sw : sw + 70],(9,9),0)
    else:
        im1h, im1w = img1.shape[:2]
        stitched_img[0:im1h, 0:im1w] = img1[0:im1h, 0:im1w]
        sw = img1.shape[1]-width-u
        stitched_img[v:v+im1h, sw:sw+im1w] = img2[0:im1h, 0:im1w]

    cv2.imshow('stitched', stitched_img)
    cv2.imwrite('stitched.jpeg', stitched_img)
    cv2.waitKey(0)


pimgtv = cv2.imread(sys.argv[1], 0)
pimgtv = np.asarray(pimgtv)
height, width = pimgtv.shape[:2]


print height, width

pimgiv = cv2.imread(sys.argv[2],0)
pimgiv = np.asarray(pimgiv)
nw = common(pimgtv, pimgiv)
print nw

pimgt = np.zeros((height, nw))
pimgi = np.zeros((height, nw))

print pimgt.shape
for i in range(height):
    for j in range(nw):
        pimgt[i][j] = pimgtv[i][width-1-nw-j] 

for i in range(height):
    for j in range(nw):
        pimgi[i][j] = pimgiv[i][j] 


width = nw

h_omega = int(height*0.5)
w_omega = int(nw*0.5)
fx = int(nw*0.25)
fy = int(height*0.25) 
print h_omega, w_omega
print fx, fy
# omega = cv2.rectangle(fx-w/2,fy-h/2,w,h);

eps = 1e-5
max_iter = 100

pgradix = np.zeros((height, width))
pgradiy = np.zeros((height, width))

n = 6
w = np.zeros((2, 3))
h = np.zeros((n, n))
ih = np.zeros((n, n))
b = np.zeros((n, 1))
delta_p = np.zeros((n, 1))
x = [0, 0, 0]                       #pt in coord frame of T
z = [0, 0, 0]                       #pt in coord frame of I

pgradix = cv2.Sobel(pimgi,-1, 1, 0);                # Gradient in X direction
# im = Image.fromarray(pgradix)
# cv2.cv.ConvertScale(im, im, 0.125);   # Normalize
pgradiy = cv2.Sobel(pimgi, -1, 0, 1);               # Gradient in Y direction
# cv2.ConvertScale(pgradiy, pgradiy, 0.125);    # Normalize

p = np.zeros((n, 1))                    #parameters

it = 0
while(it < max_iter):
    it += 1
    init_warp(w, p)
    h = np.zeros((n, n))
    b = np.zeros((1, n))

    u = 0           #pixel coordinates in frame T
    v = 0
    pixel_count = 0


    for i in range(w_omega):        
        u = i + fx
        for j in range(h_omega):
            v = j + fy
            x = [u, v, 1]
            z = np.dot(w,x)
            print w
            print x
            u2i = int(z[0])
            v2i = int(z[1])             
        
            if(u <= 0 or u > width or v <= 0 or v > height):
                print u,v
                sys.quit
            
            if(u2i >= 0 and u2i < width and v2i >= 0 and v2i < height):

                print pgradix.shape

                ix = pgradix[v2i][u2i]
                iy = pgradiy[v2i][u2i]

                stdesc = [u*ix , u*iy, v*ix, v*iy, ix, iy]
                stdesc = np.asarray(stdesc)

                i2 = pimgi[v2i][u2i]

                diff = int(pimgt[v][u]) - int(i2)   
                
                b += np.multiply(diff,stdesc)
                h += stdesc[:,None]*stdesc

    try:
        delta_p = np.dot(b,linalg.inv(h))
    except: continue

    p += delta_p.T

    if((np.abs(p) < eps).any()):
        print 'yes'
        break



x = [nw/2, height/2, 1]
z = np.dot(w,x)
print w
u2i = int(z[0])
v2i = int(z[1]) 
u = u2i - x[0]
v = v2i - x[1]
print u, v, nw
stitching(pimgtv, pimgiv, u, -1*v, nw)
print x[0], x[1]
print u2i, v2i










