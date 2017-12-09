
---

# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Estimate a bounding box for vehicles detected.
* Run this pipeline on a video stream


[//]: # (Image References)
[image1]: ./output_images/pipeline/vehicle.jpg "vehicle"
[image2]: ./output_images/pipeline/non_vehicle.jpg "non_vehicle"
[image3]: ./output_images/pipeline/boxes.jpg "boxes"
[image4]: ./output_images/pipeline/detections.jpg "detected" 
[image5]: ./output_images/pipeline/Cars "cars" 
[image6]: ./output_images/test/test1.jpg 
[image7]: ./output_images/test/test3.jpg 
[image8]: ./output_images/pipeline/pipeline.jpg 
[video1]: ./project_video_output.mp4

---

## Histogram of Oriented Gradients (HOG)

I defined the `f_hog` function to extract the hog feature (with the `skimage.hog()`) for every image.
I set `orientations = 11`, `pixels_per_cell = 16`, and `cells_per_block = 2` and used the `yuv`

![alt text][image1]

![alt text][image2]

## Classifier

I processed the vehicle/non-vehicle dataset(feature_extraction and standardization); then I split in train and cross-validation set using `train_test_split` with stratification.
After this I trained a support vector machine with `rbf` kernel
obtaining a `0.991` of accuracy with 3-fold cv.   

## Sliding Window Search

I decided to search for different scales in the bottom part of the image: I tried, defining `compute_windows` and `collect_windows`, to compute the HOG feature for every windows, but it resulted to be a slow method; so in the `find_subimages_boxes_features` I computed one time the features sub-sampling for every windows.
Here and example of the windows search space:

![alt text][image3]

## Bounding boxes

From the positive detections I created a `heatmap` and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify vehicles and I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image4]

## Result

I defined my pipeline: extract the feature vector, compute the prediction for vehicle detection, define the bounding boxes using; finally I obtained:

![alt text][image6]

![alt text][image8]

## Video Implementation

For the video stream I implemented two method for false positive detection into the `Detector()` class: the first is the thresholding on the heatmap; the second is that, if in the actual frame we identify a different number of cars than in the last frame, we don't identify a box and we wait for a new detection in the next frame: if in the next frame the classifier identifies again the same number of vehicles I consider a true positive detection 

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

My approach is not robust and there are problem when the vehicle is on the right of the image; I see some problem also when a new vehicle enters in the image.
In general I have some false negative. And I could improve my classifier.