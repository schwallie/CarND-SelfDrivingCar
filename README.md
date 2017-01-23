# Overview

![alt text](https://i.imgur.com/wTHwzwp.gifv "It works whaattt")

The [Udacity Self Driving Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) Term 1 is mostly about Deep Learning using Tensorflow and Keras to build Convolutional Neural Nets (CNNs).

In Project 3, we are given a simulator that we need to use to gather data and then train a CNN to drive itself around the track. What we're really doing is simple ;), we're training a network to predict a steering angle based off an image.

It took me way longer than I hoped to do this, but I finally figured it out and it's now successfully staying on the track!!


# Getting the simulator to run:

Clone this repo, then open a terminal in the main folder, and run:

```python
python drive.py
```

Open up the driving simulator program and it should work from there! (The driving simulator would need to be given to you from Udacity)

# General data tidbits

The data I used was from Udacity, as they had a more stable dataset to work with. I had issues using my keyboard to get realistic images to train the model with.