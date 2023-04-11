# RTSP example
# "rtsp://zephyr.rtsp.stream/pattern?streamKey=79d685bd2d7ebea91e8c2bac5543e69a"
# "rtsp://zephyr.rtsp.stream/movie?streamKey=4a49ce5a8a03fb76de26daaa2d3eda00"

from flask import Flask, render_template, Response, request
import cv2
import datetime
import time
import os
import numpy as np
from threading import Thread, Lock
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

NUMBER_OF_CAMERA = 4

params = {
    'rec_frame' : None,
    'outputFrame' : None,
    'grey' : False,
    'face' : False, 
    'rec' : False,
    'switch' : True,
    'camera' : cv2.VideoCapture(0),
    'lock' : Lock(),
    'address' : [],
    'cap' : []
}

# instatiate flask app
app = Flask(__name__)

# Load pretrained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

params['net'] = cv2.dnn.readNetFromCaffe(str(ROOT / 'saved_model/deploy.prototxt.txt'),
                    str(ROOT / 'saved_model/res10_300x300_ssd_iter_140000.caffemodel'))


def connectRTSP(source):
    if len(params['cap']) < NUMBER_OF_CAMERA:
        params['address'].append(source)
        params['cap'].append(cv2.VideoCapture(source))
    else:
        params['address'][len(params['cap']) % NUMBER_OF_CAMERA] = source
        params['cap'][len(params['cap']) % NUMBER_OF_CAMERA] = cv2.VideoCapture(source)
    print(
        f"Connected to RTSP camera as 'Camera {(len(params['cap']) - 1) % NUMBER_OF_CAMERA}")
    print(f"Address: {source}")
    time.sleep(2.0)


def record(out):
    while params['rec']:
        time.sleep(0.05)
        out.write(params['rec_frame'])


def detect_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    params['net'].setInput(blob)
    detections = params['net'].forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    startX, startY, endX, endY = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        h, w = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def generateLocalCamera():  # generate frame by frame from camera
    while True:
        success, frame = params['camera'].read()
        if success:
            if params['face']:
                frame = detect_face(frame)
            if params['grey']:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if params['rec']:
                params['rec_frame'] = frame
                frame = cv2.putText(
                    cv2.flip(frame, 1),
                    "Recording...", (0, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generateRTSPCamera(camID):
    def generate():
        if camID < len(params['cap']) and params['cap'][camID].isOpened():
            # loop over frames from the output stream
            while True:
                ret_val, frame = params['cap'][camID].read()
                if not ret_val:
                    continue
                if frame.shape:
                    frame = cv2.resize(frame, (640, 360))
                    with params['lock']:
                        outputFrame = frame.copy()

                # wait until the lock is acquired
                with params['lock']:
                    # check if the output frame is available, otherwise skip
                    # the iteration of the loop
                    if outputFrame is None:
                        continue

                    # encode the frame in JPEG format
                    (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

                    # ensure the frame was successfully encoded
                    if not flag:
                        continue

                # yield the output frame in the byte format
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                      bytearray(encodedImage) + b'\r\n')
        else:
            print(f'Camera {camID} open failed')
    return generate


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    if request.method == 'POST':
        if request.form.get('grey') == 'Grey': params['grey'] = not params['grey']
        if request.form.get('face') == 'Face Only': params['face'] = not params['face']
        if params['face']: time.sleep(4)
        if request.form.get('stop') == 'Webcam':
            if params['switch'] == 1:
                params['camera'].release()
                cv2.destroyAllWindows()
            else:
                params['camera'] = cv2.VideoCapture(0)
            params['switch'] = not params['switch']
        elif request.form.get('rec') == 'Start/Stop Recording':
            params['rec'] = not params['rec']
            if params['rec']:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                params['out'] = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))

                thread = Thread(target=record, args=[params['out'], ])
                thread.start()
            elif params['rec'] == False:
                params['out'].release()
        elif request.form.get('rtsp'):
            connectRTSP(request.form['rtsp'])

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generateLocalCamera(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/<camID>')
def video(camID):
    return Response(generateRTSPCamera(camID=int(camID))(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

for c in params['cap']: c.release()
params['camera'].release()
cv2.destroyAllWindows()
