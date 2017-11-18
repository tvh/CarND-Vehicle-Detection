**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The features are extracted in the `prepare` function.
I used `orientations=9`, `pixels_per_cell=16`, `cells_per_block=2` and all the channels of the YUV colorspace.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried different colorspaces (RGB, HLS, YUV) and YUV gave me the best results.
beyond 9 orientations I wasn't getting any more accuracy from the classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a `GradientBoostingClassifier` for this project. With this I was able to get 98.5% accuracy for the validation set. 
With a `LinearSVC` I was only able to get 96%.
The code for this, along with some data exploration, is in [Classifier.ipynb](./Classifier.ipynb).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To find cars in the image, I exhausively search the lower half of each image.
I do this for 1x, 1.5x, 2x and 3x scaling.
The scales I used were just scales that would work well when resizing the image and seemed to work okay.
3x gets to about the maximum size of any car in the video so I stopped there.

The search pattern has a 75% overlap in both x and y direction.
I tried using less of an overlap, but wasn't getting a good result from it.

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

Before tackling the video I tested my pipeline on the provided test images.
The results are saved to `output_images`.

In the first iteration the classifier hat lots of false positives.
This was with a `LinearSVC` and a coarser search pattern.
To mitigate this, I increased the overlap to 75% and used a `GradientBoostingClassifier`.

To exclude the false positives, I only consider those regions which have more than 2 overlapping matches.

### Video Implementation

#### 1. Provide a link to your final video output.
Here's a [link to my video result](./project_out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Like in the implementation for single images, I only consider regions which have multiple overlapping matches.
In addition to this I also use the heatmaps from multiple consecutive images to get the final heatmap.

The code for this part is in `annotate_video`.

### Discussion

Using the sliding window approach I found it quite hard to get accurate bounding boxes. 
The classifier might detect (correctly) that the window contains a car, but the car is not centered, the bounding box can easily get too large.
On the other hand you need to have a very fine search pattern to identify far away cars correctly.

Using this approach it is also almost impossible to seperate 2 overlapping bounding boxes.
This happens in the project video where the white car gets overtaken by the black car.
One could use color in this case to seperate them out, but this doesn't generalise well.

Another drawback of this approach is performance.
Using a sliding window approach is inherently slow and there is always the tradeoff between speed and accuracy.
I could have also made more of the computation shared (scaling, extracting hog features), but it would still be hard to get realtime performance.

If done again, I think I would use a SSD instead.
