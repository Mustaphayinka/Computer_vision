from flask import Flask, render_template,Response
import cv2
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import os

# defining face detector
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor = 0.6


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        # self.video = VideoStream(src=0).start()
    
    def __del__(self):
        self.video.release()
        # cv2.VideoCapture(0).release()

    def get_frame(self):   
        success, image = self.video.read()
        image = cv2.resize(image, None, fx = ds_factor, fy = ds_factor, 
        interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        # for (x,y,w,h) in face_rects:
        #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            # break
        # success, image = cv2.VideoCapture(0).read()
        image, num_mask, num_no_mask = process(image)
        
        ret, jpeg = cv2.imencode('.jpeg', image)
        return jpeg.tobytes()


prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")


# Initialize flask app
app = Flask(__name__)


def predict(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.4:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)


def process(frame):
	frame = imutils.resize(frame, width=1080)
	(locs, preds) = predict(frame, faceNet, maskNet)
	num_mask = 0
	num_no_mask = 0
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		if mask > withoutMask:
			label = "MASKED FACE"
			num_mask += 1
		else:
			label = "NO MASK"
			num_no_mask += 1
		
		if label == "MASKED FACE":
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	return frame, num_mask, num_no_mask




@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame() 
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




@app.route('/video')
def video():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug = True)


   