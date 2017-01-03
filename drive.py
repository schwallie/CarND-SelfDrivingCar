import argparse
import base64
from io import BytesIO
import cv2
import eventlet.wsgi
import numpy as np
import socketio
import config
import tensorflow as tf
from PIL import Image
from flask import Flask

tf.python.control_flow_ops = tf

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    prev_steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    # image = np.float32(cv2.resize(cv2.imread(image, 1)[32:140, 0:320], (200, 66))) / 255.0
    image_array = np.asarray(image)
    image_array = config.return_image(image_array)
    image_array = image_array/255. - 0.5
    cv2.imwrite('Saved_Img.png', image_array)
    transformed_image_array = image_array[None, :, :, :]
    # transformed_image_array = cv2.resize(transformed_image_array, (200, 66))
    # print(transformed_image_array.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(saved_model.predict(transformed_image_array, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    print('Angle: {0}'.format(round(steering_angle, 4)))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    import model
    saved_model = model.load_saved_model('model.h5')
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
