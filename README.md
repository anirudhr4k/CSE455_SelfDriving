# CSE 455: Final Project

# Abstract

# Problem
One of the most important and relevant applications of computer vision is in the context of Autonomous Driving. As we keep making progress in the fields of Artificial Intelligence and Machine Learning, Self-Driving Cars are no longer just a part of futuristic fiction. Many companies around the world such as Waymo, Cruise, and Tesla are investing hundreds and thousands of dollars into autonomous driving research and have begun rolling out fully- or semi-autonomous driving agents. While there are several approaches to implementing autonomous driving, one such approach is mounting an RGB-D camera atop a car to detect pedestrians and other vehicles in the car's vicinity so as to not only estimate the car's current state with respect to the map but also to classify possible obstacles that the car will have to watch out for or circumnavigate in the future. We plan to use vehicular imaging data and object detection data in a slightly unorthodox way to implement lane detection in the context of planning.

The primary goal of our project will be to use the techniques we have learned in class as well as other tools that we will research online to 1) detect which lane the car is driving in, 2) detect obstacles on the road and which lanes said obstacles will block, and 3) provide driving suggestions based on other potentially free lanes. While we would love to implement all 3 goals, the goals are listed in order of importance to our final project. This means that we will definitely achieve goal number one, most probably achieve goal number two, and – if time permits – also try to achieve goal number three. Specifically, we plan on using the Udacity Self-Driving Car DatasetLinks to an external site., The Berkeley Deep Drive (BDD110K) DatasetLinks to an external site., and the Comma2k19Links to an external site. datasets to implement our lane-detection and object-detection algorithm. One computer vision technique we will use is the canny detector, a popular algorithm for edge detection. It uses a series of steps to find the boundaries of objects in an image. The steps are noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding. These steps will help us meet goal 1. Our next step will be to detect cars in the scene, likely using OpenCV and haar cascade classifiers. Finally, we will try to convert our scene into a position grid so as to determine the coordinates of all of the cars in the scene with respect to the positions of the lane so we can determine which lanes contain cars. This will help us solve goals 2 and 3.

# Datasets

# Methodology

# Results

## Video

# Conclusion

# Glossary
