import numpy


import numpy as np
import numpy.linalg as linalg
import sys
import cv2, cv
import math
import scipy.stats
from PIL import Image


fx, fy = -1, -1

def get_feature(event,x,y,flags,param):
    global fx, fy   
    if event == cv2.EVENT_LBUTTONDBLCLK:
        refPt = [(x, y)]
        fx, fy = x, y
        # cv2.circle(pimgi,(x,y),10,(255,0,0),-1)



def init_warp(w, p):
    w[0][0] = 1 + p[0]
    w[0][1] = p[2]
    w[0][2] = p[4]
    w[1][0] = p[1]
    w[1][1] = 1 + p[3]
    w[1][2] = p[5]



cap = cv2.VideoCapture(sys.argv[1])

ret, frame = cap.read()

height, width = frame.shape[:2]

pimgt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
pimgi = pimgt

cv2.namedWindow('image')
cv2.setMouseCallback('image', get_feature)
while(1):
    cv2.imshow('image',pimgi)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    if [fx,fy] != [-1,-1]:
        break
print fx,fy


counter = 0

while(1):

    if counter != 0:
        pimgt = pimgi

    counter += 1
    ret, frame = cap.read()
    if not ret:
        break


    print counter
  
    pimgi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image2',frame)


    h_omega = 15
    w_omega = 15
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

    # pimgi = cv2.GaussianBlur(pimgi,(5,5),0)
    pgradix = cv2.Sobel(pimgi,-1, 1, 0);                # Gradient in X direction
    # im = cv.fromarray(pgradix)
    # cv2.cv.ConvertScale(im, im, 0.75);   # Normalize
    # pgradix = numpy.asarray(im)
  
    pgradiy = cv2.Sobel(pimgi, -1, 0, 1);               # Gradient in Y direction
    # im = cv.fromarray(pgradix)
    # cv2.cv.ConvertScale(im, im, 0.75);    # Normalize
    # pgradiy = numpy.asarray(im)

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
            u = i + fx-w_omega/2
            # print u
            for j in range(h_omega):
                v = j + fy-h_omega/2
                # print v
                x = [u, v, 1]
                z = np.dot(w,x)
                u2i = int(z[0])
                v2i = int(z[1]) 
                
            
                if(u <= 0 or u > width or v <= 0 or v > height):
                    print u,v
                    sys.quit
                
                if(u2i >= 0 and u2i < width and v2i >= 0 and v2i < height):

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
        # print p 

        if((np.abs(p) < eps).any()):
            print 'yes'
            break

    x = [fx, fy, 1]
    z = np.dot(w,x)
    print w
    u2i = int(z[0])
    v2i = int(z[1]) 
    # print fx, fy
    # print u2i, v2i
    fx = u2i
    fy = v2i
    # cv2.circle(frame,(fx,fy),5,(0,255,0),-1)
    cv2.circle(frame,(u2i,v2i),5,(0,255,0),-1)
    cv2.imshow('image1',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break