# CSE 455: Final Project

# Problem

One of the most important and relevant applications of computer vision is in the context of Autonomous Driving. As we keep making progress in the fields of Artificial Intelligence and Machine Learning, Self-Driving Cars are no longer just a part of futuristic fiction. Many companies around the world such as Waymo, Cruise, and Tesla are investing hundreds and thousands of dollars into autonomous driving research and have begun rolling out fully- or semi-autonomous driving agents. While there are several approaches to implementing autonomous driving, one such approach is mounting an RGB-D camera atop a car to detect pedestrians and other vehicles in the car's vicinity so as to not only estimate the car's current state with respect to the map but also to classify possible obstacles that the car will have to watch out for or circumnavigate in the future. We plan to use vehicular imaging data and object detection data in a slightly unorthodox way to implement lane detection in the context of planning. The primary problem that we will be solving is detecting the lane in which lane the car is currently moving in and where any other cars in the vicinity are.

# Datasets

 The dataset that we used to train our model for this project was the [Udacity Self-Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car).

The dataset consisted of ~8000 images of cars and ~8000 images of non-cars that we used to train a classifier which – given an image – could detect whether or not there was a car in the image.

# Set up

The set up for this project was fairly minimal. The Udacity dataset contained preprocessed and labelled images.

The set up was even simpler for lane tracking, as that was done purely through edge detection. All we needed was OpenCV installed and a video to test on.

# Methodology

## Lane Tracking

Lane tracking was done through edge detection. It followed a fairly standard Hough transform workflow in which each image was preprocessed, edge detected, applied to the Hough lines transform and the lines were overlaid on the original image.

The Hough Line transform is a default method provided by OpenCV. It’s a method that is used to detect straight lines. Since lanes are marked by straight lines, the idea behind this workflow is that we simply apply this standard transform to an input image. More information about this transform can be found [here](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html).

However, there are two main problems with this:
1. Dashed lanes they will form many lines when we might only want one per side
2. The image can contain noise in the form of other objects with straight lines

Thankfully, there is an easy fix to both of the issues. To fix the multiple lines with dashed lane markers, all we really need to do is to form a “master line” that is calculated by taking the average of the slopes of all the lines on one side while using the most extreme points as the two end points.

To fix the other problem, all we would need to do is remove as many of the distractions as possible. This is done by simply masking the image to contain only a sliver in the middle, where the lane markings remain.

Overall, the process in which the lanes were identified are:
1. Blur the image (in an attempt to get rid of errant lines)
2. Mask the image (to remove the useless bits of information)
3. Perform Canny edge detection (to get just the straight lines from the image)
4. Perform the Hough line transform
5. Clean up the lines

After applying the canny, we got the:
![Canny](canny.gif)

After the masking, we get:
![Masked](masked_canny.gif)

After applying the Hough line transform and adding the lines to the original image, we get the [video](https://youtu.be/UmRVmtpJfyk)

Details of how this was applied can be found in the Github, linked above.

Applying both of these methods works remarkably well to detect lanes in most conditions. There are, however, a few limitations that we will mention later.

In performing research for this project, we have found that this methodology seems to be fairly standard in detecting lanes (at least in cases where machine learning was not used to detect lanes).


## Object Detection

When it came to the object detection aspect of this project, we were initially unsure of what approach would work best. We started off by using a haar cascade classifier to detect whether or not an image contained a car. However, even though this approach worked on simple images, it did not work too well in images that had multiple cars.
We thus decided to create a machine learning model and train it on the dataset mentioned above. Initially, we decided to use a Support Vector Machine and trained it on the Udacity Car dataset. However, even though the SVM performed very well on the test data (it got up to 99% accuracy) it performed very poorly on the input data, likely because during training, the model had overfit to the training data. Tuning the parameters for the SVM did show a little bit of improvement but not too much.
After doing some research, we found that for classification problems like the one we were trying to solve, a neural network was usually the best option. We then decided to experiment with the architecture of the neural network, and settled on the following:

```py
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

We split the data we had into training, validation, and testing data and trained our model. We got 97% accuracy on the test data, which we were pretty happy with. The weights that we got after training our model can be found [here](https://drive.google.com/file/d/10eQ6x4AoKCAbmmtHoBpWQkfoUU4h6ssP/view).
To apply the model to the actual input data, we decided to use a sliding window approach where each window was put through the model to see if the window contained a car or not. We experimented with several window sizes and found that a window size of 75 worked best, both for distant and nearby cars.

# Results

Lane tracking seems to be fairly accurate much of the time.

There are a few cases in which lane tracking fails, primarily at points in which there are discrepancies within the road itself, such as cracks, change in pavement type, and so on and so forth. There’s also a few frames in which the lane tracking algorithm failed altogether, due to either failing to detect a line for some odd reason or from failing to find an “average” line.

You can find a video [here](https://youtu.be/UmRVmtpJfyk)

### Video

The results for the object detection aspect of our project were pretty good as well. In addition to getting 97% accuracy on the test data, our model also did pretty well in identifying cars in the input images. The results were slightly poorer for frames in which there are a lot of shadows and extensive tree cover, but overall, the model does a fairly good job of classifying car vs non-car frames. Because we did not get enough time to merge the bounding boxes for overlapping windows, the final results appear closer to a heatmap of where cars actually are as opposed to perfect bounding boxes for those cars but we still find our model's results very satisfactory.

### Pictures (Combined Lane Tracking and Object Detection)

[TODO: insert pictures]

## Limitations/Errors

There were several limitations that we found in implementing this project.

#### Lane Tracking
- Failure to detect curves
- Prone to getting confused by random lines detected by the edge detection algorithm
#### Object Detection
- While the sliding window approach that we used made our search very thorough, it slowed down the object detection process.
- Our decision to use a neural network meant that we got very good accuracy, but our predictions took a long time. This meant that it was infeasible for us to apply the object detection pipeline to constant video streams.
- The model did not perform too well on cars that were very far away or very close, likely because we used a fixed window size when doing our search. Our initial approach was to use different window sizes but we realized that using multiple window sizes was slowing down our bounding box detection/drawing considerably which is why we decided to stick with a single window size.

## Next Steps

One goal that we had for this project was to create an overall workflow that worked to do both lane tracking and object detection and to use the conjunction of this data to identify whether there was an object in the current lane. While we were able to get the workflows to work together, we ran out of time to 

### Lane Tracking

The main next steps for lane tracking is to include functionality to detect curves in the road and to further tune the processing in order to get rid of unwanted noise/consistently lane track. 

While we are not certain on how to implement curves, we have an idea on how to detect curves. If we were able to instead get a top-down view of the road, it would be much easier to detect the edges of a curved lane with Canny and to fit a line along this instead. It may even be possible with the video we used for this project by simply doing a perspective transform.

### Object Detection
