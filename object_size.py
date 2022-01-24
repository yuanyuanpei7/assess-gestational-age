# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# load the image, convert it to grayscale, and blur it slightly
#扫描当前目录所有图片
for f in os.listdir('./'):
    if f.split('.')[-1]=='png' or f.split('.')[-1]=='jpg' or f.split('.')[-1]=='bmp' :
        print(f)
        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = 49.412 #49个像素1cm
        orig = image.copy()
        # loop over the contours individually
        for c in cnts:
        	# if the contour is not sufficiently large, ignore it
        	if cv2.contourArea(c) < 100:
        		continue
        
        	# compute the rotated bounding box of the contour
        	
        	# orig = image 可将所有结果显示，同时将最后两行显示的代码缩进可以一次性显示最终结果
        	box = cv2.minAreaRect(c)
        	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        	box = np.array(box, dtype="int")
        
        	# order the points in the contour such that they appear
        	# in top-left, top-right, bottom-right, and bottom-left
        	# order, then draw the outline of the rotated bounding
        	# box
        	box = perspective.order_points(box)
        	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        
        	# loop over the original points and draw them
        	for (x, y) in box:
        		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        	# unpack the ordered bounding box, then compute the midpoint
        	# between the top-left and top-right coordinates, followed by
        	# the midpoint between bottom-left and bottom-right coordinates
        	(tl, tr, br, bl) = box
        	(tltrX, tltrY) = midpoint(tl, tr)
        	(blbrX, blbrY) = midpoint(bl, br)
        
        	# compute the midpoint between the top-left and top-right points,
        	# followed by the midpoint between the top-righ and bottom-right
        	(tlblX, tlblY) = midpoint(tl, bl)
        	(trbrX, trbrY) = midpoint(tr, br)
        
        	# draw the midpoints on the image
        	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        	# draw lines between the midpoints
        	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        		(255, 0, 255), 2)
        	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        		(255, 0, 255), 2)
        
        	# compute the Euclidean distance between the midpoints
        	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        	print(tltrX, tltrY,blbrX, blbrY)
        	# if the pixels per metric has not been initialized, then
        	# compute it as the ratio of pixels to supplied metric
        	# (in this case, inches)
        	if pixelsPerMetric is None:
        		pixelsPerMetric = dB 
          
        	# compute the size of the object
        	dimA = dA / pixelsPerMetric
        	dimB = dB / pixelsPerMetric
        
        	# draw the object sizes on the image

        	weeks=((dimA+dimB)/2+2.54)/0.7
        	cv2.putText(orig, "{:.1f}CM".format(dimA),(int(tltrX + 50), int(tltrY - 50)), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255, 255), 2)
        	cv2.putText(orig, "{:.1f}CM".format(dimB),(int(trbrX + 50), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255, 255), 2)
        	cv2.putText(orig, "{:.1f} Weeks".format(weeks),(80, 150), cv2.FONT_HERSHEY_SIMPLEX,1.5, (255, 255, 255), 2)
        if not os.path.exists('new'):
            os.makedirs('new')        
        	# show the output image
        cv2.imwrite('./new/'+f, orig)
        
