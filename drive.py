import base64
from io import BytesIO
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
    image_array = np.asarray(image)
    # Preprocessing
    image_array = config.return_image(image_array)
    image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(saved_model.predict(image_array, batch_size=1)) * config.STEERING_ADJUSTMENT
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if abs(steering_angle) > .07:
        throttle = .01
    else:
        throttle = config.AUTONOMOUS_THROTTLE
    if abs(steering_angle) > .07:
        steering_angle = steering_angle # * 1.3
    print('Angle: {0}, Throttle: {1}'.format(round(steering_angle, 4), throttle))
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
    import config
    saved_model = model.load_saved_model('model.h5', model=model.steering_net())
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
