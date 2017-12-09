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

def find_subimages_boxes_features(img, scale, ystart = 400, ystop = 656, window=64, pix_per_cell=16, cell_per_block=2, orient=11, 
                       cells_per_step = 2):
    
    #img = img.astype(np.float32)/255
    
    img_processed = img[ystart:ystop,:,:]
    img_processed = color_space(img_processed, "yuv")
    if scale != 1:
        n, m, k = img_processed.shape
        img_processed = cv2.resize(img_processed, (np.int(m/scale), np.int(n/scale)))
        
    ch1 = img_processed[:,:,0]
    ch2 = img_processed[:,:,1]
    ch3 = img_processed[:,:,2]
    
    nc, mc = ch1.shape
    # Define blocks and steps as above
    n_blocks_x = (mc // pix_per_cell) - cell_per_block + 1
    n_blocks_y = (nc // pix_per_cell) - cell_per_block + 1 
    n_feat_block = orient*cell_per_block**2
    
    n_blocks_window = (window // pix_per_cell) - cell_per_block + 1
    n_steps_x = (n_blocks_x - n_blocks_window) // cells_per_step
    n_steps_y = (n_blocks_y - n_blocks_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = f_hog(ch1)
    hog2 = f_hog(ch2)
    hog3 = f_hog(ch3)
    
    list_boxes=[]
    #list_subimages=[]
    list_features=[]
    
    for xb in range(n_steps_x):
        for yb in range(n_steps_y):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+n_blocks_window, xpos:xpos+n_blocks_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+n_blocks_window, xpos:xpos+n_blocks_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+n_blocks_window, xpos:xpos+n_blocks_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            subimg = cv2.resize(img_processed[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # convert
            xleft_box = np.int(xleft*scale)
            ytop_box = np.int(ytop*scale)
            win_box = np.int(window*scale)
            bottom_left=(xleft_box, ytop_box+win_box+ystart)
            upper_right=(xleft_box+win_box, ytop_box+ystart )
            
            list_features.append(hog_features)
            list_boxes.append((bottom_left, upper_right))
            ##list_subimages.append(subimg)
    
    features=np.array(list_features)
    features=scaler.transform(features)
            
    return features, list_boxes

def detection_image(img, scale, flag):
    features, list_boxes=find_subimages_boxes_features(img, scale=scale)
    
    prediction=clf.predict(features)
    index_detections=list(np.where(prediction==1)[0])
    
    detections=[list_boxes[ix] for ix in index_detections]
    
    if flag:
        for d in detections:
            bottom_left=d[0]
            upper_right=d[1]
            z=cv2.rectangle(img,bottom_left,upper_right,(0,0,255), 6)
    return detections

def detection_multiscale(img, scales, flag=False):
    detections=[]
    for s in scales:
        detection=detection_image(img, scale=s, flag=flag)
        detections.extend(detection)
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

def pipeline(img, scales=[1.0, 1.6], flag=False):
    image=img.copy()
    detections=detection_multiscale(image, scales)
    heatmap=add_heat(image, detections,t=1)
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