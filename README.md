[gifdrive]: https://github.com/schwallie/CarND-SelfDrivingCar/blob/master/assets/GifRecording.gif "Self Driving Car"
[datahist]: https://github.com/schwallie/CarND-SelfDrivingCar/blob/master/assets/DataHist.png "Datahist"
[model]: https://github.com/schwallie/CarND-SelfDrivingCar/blob/master/assets/Model.png "Model"

# Overview

![driving car gif][gifdrive]

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

The main thing I noticed with the data was that on each side of 0 (left or right driving), each bucket isn't totally even. So I wanted to make sure to even out both the sides so the model didn't generalize this and try to go left or right more often than it should.

![datahist][datahist]


# Model

I used the model that comma.ai open sourced, but I'm pretty sure the NVIDIA model (also contained in the repo) would work well now that I've figured out augmentations better.

![model][model]


I used dropouts between the layers to lower problems with overfitting the model, and I also used data augmentation to avoid overfitting. Specifically,

```python
    # Augment brightness so it can handle both night and day
    image = augment_brightness_camera_images(image)
    trans = np.random.random()
    if trans < .2:
        # 20% of the time, return the original image
        return return_image(image), steering
    trans = np.random.random()
    if trans > .3:
        # Flip the image around center 70% of the time
        steering *= -1
        image = cv2.flip(image, 1)
    trans = np.random.random()
    if trans > .5:
        # Translate 50% of the images
        image, steering = trans_image(image, steering, 150)
    trans = np.random.random()
    if trans > .8:
        # 20% of the time, add a little jitter to the steering to help with 0 steering angles
        steering += np.random.uniform(-1, 1) / 60
```

I used the comma.ai model, a batch size of 128, and an AdamOptimizer. I found a lower learning rate to work much better for my model, so that we didn't reach a false minimum error.

Here are my main takeaways from my work:
1. Do not pre-generate the images. I originally made something that generated the images (translations, flips, etc) and then just read those images for training. This didn't work well. I think it's because I basically did `Original Image * CHOICE([Flip, Translation, Brightness])`, but you need to just do the translations on the fly to really give the model enough images to train on
2. Consider large numbers of EPOCHS. When using dropouts and translations, I wasn't overly worried about overfitting. My late EPOCH versions worked best
3. Do all the augmentations above, and make sure it varies. Keep your model guessing/learning
4. 128 mini-batches worked better for me than anything larger
5. Ask a lot in Slack and the forums!

