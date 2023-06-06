# CSE 455: Final Project

# Problem

One of the most important and relevant applications of computer vision is in the context of Autonomous Driving. As we keep making progress in the fields of Artificial Intelligence and Machine Learning, Self-Driving Cars are no longer just a part of futuristic fiction. Many companies around the world such as Waymo, Cruise, and Tesla are investing hundreds and thousands of dollars into autonomous driving research and have begun rolling out fully- or semi-autonomous driving agents. While there are several approaches to implementing autonomous driving, one such approach is mounting an RGB-D camera atop a car to detect pedestrians and other vehicles in the car's vicinity so as to not only estimate the car's current state with respect to the map but also to classify possible obstacles that the car will have to watch out for or circumnavigate in the future. We plan to use vehicular imaging data and object detection data in a slightly unorthodox way to implement lane detection in the context of planning. The primary problem that we will be solving is the following: how can we detect which lane the car is currently moving in, and are there any other cars in the vicinity that we need to be aware of?

The primary goal of our project will be to use the techniques we have learned in class as well as other tools that we will research online to 1) detect which lane the car is driving in, 2) detect obstacles on the road and which lanes said obstacles will block, and 3) provide driving suggestions based on other potentially free lanes. While we would love to implement all 3 goals, the goals are listed in order of importance to our final project. This means that we will definitely achieve goal number one, most probably achieve goal number two, and – if time permits – also try to achieve goal number three. Specifically, we plan on using the Links to an external site., The Berkeley Deep Drive (BDD110K) DatasetLinks to an external site., and the Comma2k19Links to an external site. datasets to implement our lane-detection and object-detection algorithm. One computer vision technique we will use is the canny detector, a popular algorithm for edge detection. It uses a series of steps to find the boundaries of objects in an image. The steps are noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding. These steps will help us meet goal 1. Our next step will be to detect cars in the scene, likely using OpenCV and haar cascade classifiers. Finally, we will try to convert our scene into a position grid so as to determine the coordinates of all of the cars in the scene with respect to the positions of the lane so we can determine which lanes contain cars. This will help us solve goals 2 and 3.

# Datasets

 The dataset that we used to train our model for this project was the [Udacity Self-Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car).

The dataset consisted of ~8000 images of cars and ~8000 images of non-cars that we used to train a classifier which – given an image – could detect whether or not there was a car in the image.





# Set up

The set up for this project was fairly minimal. The Udacity dataset contained preprocessed and labelled images.

The set up was even simpler for lane tracking, as that was done purely through edge detection. All we needed was OpenCV installed and a video to test on.

# Methodology

## Lane Tracking

Lane tracking was done through edge detection. It followed a fairly standard Hough transform workflow in which each image was preprocessed, edge detected, applied to the Hough lines transform and the lines were overlaid on the original image.


## Object Detection

When it came to the object detection aspect of this project, we were initially unsure of what approach would work best. We started off by using a haar cascade classifier to detect whether or not an image contained a car. However, even though this approach worked on simple images, it did not work too well in images that had multiple cars. We thus decided to create a machine learning model and train it on the dataset mentioned above. After doing some research, we found that for classification problems like the one we were trying to solve, a neural network was usually the best option. We then decided to experiment with the architecture of the neural network, and settled on the following:
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



# Results

## Lane Tracking

### Pictures
### Video

## Object Detection

### Pictures
### Video

## Combined Workflow

# Conclusion

## Limitations
## Next Steps

# Glossary

