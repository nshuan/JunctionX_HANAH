from flask import Flask, render_template, Response, request
import cv2
import datetime
import time
import os
import numpy as np
from threading import Thread
from pathlib import Path
from overlap import overlap

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

NUMBER_OF_CAMERA = 4

BLANK = np.zeros((360, 540, 3), np.uint8)
cv2.putText(BLANK, 'No frame', (150, int(90 * 5e-3 * 360)),
            0, 2e-3 * 360, (255, 255, 255), 2)

params = {
    'rec_frame': None,
    'rec': False,
    'address': [],
    'cap': [],
    'cam_count': 0
}

blank_status = []

log_message = ""

# instatiate flask app
app = Flask(__name__)


def logger(msg):
    global log_message
    log_message = log_message + '\n' + \
        str(datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S: ")) + msg
    print(msg)
    return render_template("index.html", log_message=log_message)


def connectRTSP(source):
    if params['cam_count'] < NUMBER_OF_CAMERA:
        params['address'].append(source)
        params['cap'].append(cv2.VideoCapture(source))
    else:
        params['address'][params['cam_count'] % NUMBER_OF_CAMERA] = source
        params['cap'][params['cam_count'] %
                      NUMBER_OF_CAMERA] = cv2.VideoCapture(source)
    params['cam_count'] += 1
    logger(
        f"Assigned video stream '{source}' as 'Camera {(params['cam_count'] - 1) % NUMBER_OF_CAMERA}'")
    print(f"Address: {source}")


def record(out):
    while params['rec']:
        time.sleep(0.05)
        out.write(params['rec_frame'])


@app.route('/')
def index():
    return render_template('index.html', log_message=log_message)


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    if request.method == 'POST':
        if request.form.get('rec') == 'Start/Stop Recording':
            params['rec'] = not params['rec']
            if params['rec']:
                logger("Record start")
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                params['out'] = cv2.VideoWriter('vid_{}.avi'.format(
                    str(now).replace(":", '')), fourcc, 20.0, (640, 480))

                thread = Thread(target=record, args=[params['out'], ])
                thread.start()
            elif params['rec'] == False:
                logger("Record end")
                params['out'].release()
        elif request.form.get('clear') == 'Clear all':
            for cap in params['cap']:
                cap.release()
            params['cap'].clear()
            params['address'].clear()
        elif request.form.get('rtsp'):
            connectRTSP(request.form['rtsp'])

    elif request.method == 'GET':
        return render_template('index.html', log_message=log_message)
    return render_template('index.html', log_message=log_message)


def concat_images(img_list: list):
    """ Concat a list of image to a single image with width 1080px """
    rows = []
    if len(params['cap']) >= 2:
        overlaped, t, poly = overlap(img_list)
        img_list = [BLANK if blank_status[i]
                    else im for i, im in enumerate(overlaped)]
    for i in range(1, len(params['cap']), 2):
        row = cv2.hconcat([img_list[i-1], img_list[i]])
        rows.append(row)
    if len(params['cap']) % 2 == 1:
        row = cv2.hconcat([img_list[-1], BLANK])
        rows.append(row)
    if len(rows) > 0:
        img_list = cv2.vconcat(rows) if len(rows) > 1 else rows[0]
        if len(params['cap']) > 1:
            cv2.putText(img_list, f'{t:.3f}s', (10, int(
                10 * 5e-3 * 360)), 0, 0.6, color=(0, 0, 255), thickness=2)
            cv2.putText(img_list, f'FPS: {1/t:.1f}', (10, int(24 *
                        5e-3 * 360)), 0, 0.6, color=(0, 0, 255), thickness=2)
    else:
        img_list = BLANK
    img = cv2.imencode(".jpg", img_list)[1]
    params['rec_frame'] = img
    return img


def gen_all_camera_frames():
    """ Generate all camera frame into 1 frame """
    if len(params['cap']) == 0:
        return
    while True:
        img_list = []
        blank_status.clear()
        for i, cap in enumerate(params['cap']):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (540, 360))
                cv2.putText(frame, f'CAM {i}', (470, int(
                    12 * 5e-3 * 360)), 0, 0.6, color=(255, 0, 0), thickness=2)
                img_list.append(frame)
                blank_status.append(False)
            else:
                img_list.append(BLANK)
                blank_status.append(True)
        img = concat_images(img_list)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(img) + b'\r\n')


@app.route('/video_camera')
def video_camera():
    return Response(gen_all_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)

for c in params['cap']:
    c.release()
cv2.destroyAllWindows()
