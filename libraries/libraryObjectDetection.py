import os
import cv2
import glob
import time
import random
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label

def f_hog(channel, pix_per_cell=16, cell_per_block=2, orient=11, flag=False):
    if flag:
        features, hog_image = hog(channel, orientations=orient,
                              pixels_per_cell=(pix_per_cell, pix_per_cell), 
                              cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True, visualise=flag, feature_vector=False,
                                  block_norm="L2-Hys")
        return hog_image, features
    else:
        features = hog(channel, orientations=orient,
                      pixels_per_cell=(pix_per_cell, pix_per_cell), 
                      cells_per_block=(cell_per_block, cell_per_block),
                          transform_sqrt=True, visualise=flag, feature_vector=False,
                          block_norm="L2-Hys")
        return features

def preprocess(img, scaler, m):
    img_processed=img.copy()
    #img_copy=gaussian_blur(img_copy, 5)
    img_processed=cv2.resize(img_processed, (64, 64)) 
    img_processed=color_space(img_processed, "yuv")
    channels=[img_processed[:,:,0], img_processed[:,:,1], img_processed[:,:,2]]
    features=[]
    for c in channels:
        f = f_hog(c)
        f=f.ravel()
        features.extend(f)
    features=np.array(features[:m])
    features=np.reshape(features, newshape=(1,m))
    features=scaler.transform(features)
    return features

def compute_windows(boxes, s="s", flag=False):
    n, m, _ = img_copy.shape
    step=0.5
    k = n//2 + 30
    
    if s == "s":
        size=n//10
        n_y= (k // size)
        n_x= (m // size)
        range_y=[0.0,1.0,2.0] 
        range_x=[i for i in np.arange(5, n_x, step)]
    elif s == "m":
        size=n//8
        n_y= (k // size)
        n_x= (m // size)
        range_y=[i for i in np.arange(0, n_y, step)] 
        range_x=[i for i in np.arange(2, n_x, step)]
    elif s == "l":
        size=n//6
        n_y= (k // size)
        n_x= (m // size)
        range_y=[i for i in np.arange(0, n_y, step)] 
        range_x=[i for i in np.arange(1, n_x, step)]
        
    if s=="s":
        color=(0,0,255)
    elif s=="m":
        color=(255,0,0)
    elif s=="l":
        color=(0,255,0)
    
    for y in range_y:
        for x in range_x:
            xb, yb = int(x*size), int(k+size*(y+1))
            xu, yu = int(size*(x+1)), int(k+size*(y))
            bottom_left=(xb, yb)
            upper_right=(xu, yu)
            if flag:
                z=cv2.rectangle(img_copy,bottom_left,upper_right,color, 2)
            boxes.append((bottom_left, upper_right))
    return boxes

def collect_windows(img, flag=False):
    boxes = []
    sizes=["s", "m", "l"]
    for s in sizes:
        boxes=compute_windows(boxes, s, flag=flag)
    return boxes

def select_box(img, box):
    xb, yb = box[0]
    xu, yu = box[1]
    img_box=img[yu:yb, xb:xu]
    return img_box

def f_detection(img, boxes, m=1188, flag=False):
    img_detection=img.copy()
    detections=[]
    for box in boxes:

        img_box=select_box(img_detection, box)
        features=preprocess(img_box, scaler, m)
        p=clf.predict(features)

        if p == 1:
            detections.append(box)
    if flag:
        for d in detections:
            bottom_left=d[0]
            upper_right=d[1]
            z=cv2.rectangle(img_detection,bottom_left,upper_right,(0,0,255), 2)
        return img_detection, detections
    return detections

def add_heat(img, detections, t):
    heatmap = np.zeros_like(img[:,:,0])
    for box in detections:
        xb, yb = box[0]
        xu, yu = box[1]
        ## ((x1, y1), (x2, y2))
        heatmap[yu:yb, xb:xu] += 1
    # Return updated heatmap
    heatmap[heatmap <= t] = 0
    return heatmap

def compute_labels(heatmap, flag=False):
    labels = label(heatmap)
    if flag:
        plot_image(labels[0], cmap='gray')
        
    return labels

def compute_boxes(labels):
    boxes=[]
    n=labels[1]
    for i in range(1, n+1):
        nonzero = (labels[0] == i).nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        bottom_left = (np.min(nonzero_x), np.min(nonzero_y))
        upper_right = (np.max(nonzero_x), np.max(nonzero_y))
        box = (bottom_left, upper_right)
        boxes.append(box)
    return boxes

 def draw_boxes(img, boxes):
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
    return img

def pipeline(img, flag=False):
    image=img.copy()
    windows=collect_windows(image)
    detections=f_detection(image,windows)
    heatmap=add_heat(img, detections t=0)
    labels=compute_labels(heatmap)
    boxes=compute_boxes(labels)
    
    if flag:
        draw_img = draw_boxes(image, boxes)
        return draw_img, boxes
    
    return boxes

def process_image(image):
    new_image = image.copy()
    boxes=pipeline(new_image)
    
    # detector=Detector()
    detector.update(boxes)
    average_boxes=detector.get_boxes()
    
    res = draw_boxes(image, boxes)
    return res