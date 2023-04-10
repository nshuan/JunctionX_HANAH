from flask import Flask, render_template, Response, request
import cv2
import threading
import datetime, time, os
import numpy as np
from threading import Thread

global capture, rec_frame, grey, switch, face, rec, out
outputFrame = None
capture = 0
grey = 0
face = 0
switch = 1
rec = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

lock = threading.Lock()
camera = cv2.VideoCapture(0)

# RTSP example
# "rtsp://zephyr.rtsp.stream/pattern?streamKey=79d685bd2d7ebea91e8c2bac5543e69a"
# "rtsp://zephyr.rtsp.stream/movie?streamKey=4a49ce5a8a03fb76de26daaa2d3eda00"

NUMBER_OF_CAMERA = 4

address = []
cap = []
# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

def connectRTSP(source):
    if len(cap) < NUMBER_OF_CAMERA:
        address.append(source)
        cap.append(cv2.VideoCapture(source))
    else:
        address[len(cap) % NUMBER_OF_CAMERA] = source
        cap[len(cap) % NUMBER_OF_CAMERA] = cv2.VideoCapture(source)
    print(f"Connected to RTSP camera as 'Camera {(len(cap) - 1) % NUMBER_OF_CAMERA}' \nAddress: {source}")
    time.sleep(2.0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame

def generateLocalCamera():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if(face):
                frame = detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(capture):
                capture = 0
                now = datetime.datetime.now()
                p = os.path.sep.join(
                    ['shots', "shot_{}.png".format(str(now).replace(":", ''))])
                cv2.imwrite(p, frame)
            if(rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(
                    frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generateRTSPCamera(camID):
    def generate():
    # grab global references to the output frame and lock variables
        global lock
        if camID < len(cap) and cap[camID].isOpened():
            # loop over frames from the output stream
            while True:
                ret_val, frame = cap[camID].read()
                if not ret_val:
                    continue
                if frame.shape:
                    frame = cv2.resize(frame, (640,360))
                    with lock:
                        outputFrame = frame.copy()

                # wait until the lock is acquired
                with lock:
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

@app.route('/video0')
def video0():
    return Response(generateLocalCamera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video1')
def video1():
    return Response(generateRTSPCamera(0)(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    return Response(generateRTSPCamera(1)(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video3')
def video3():
    return Response(generateRTSPCamera(2)(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, capture, grey, face, rec, out
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = 1
        elif request.form.get('grey') == 'Grey':
            grey = not grey
        elif request.form.get('face') == 'Face Only':
            face = not face
            if face:
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            rec = not rec
            if rec:
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))

                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif rec == False:
                out.release()
        elif request.form.get('rtsp'):
            connectRTSP(request.form['rtsp'])

    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
 
cap.release()
camera.release()
cv2.destroyAllWindows()