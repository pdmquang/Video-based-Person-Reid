# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import os
import numpy as np
import argparse
import time
import cv2
import pickle

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('reid_model/custom_softmax_pretrain.h5')
with open(r"reid_model/label_mapping.pickle", "rb") as input_file:
	labels_dict = pickle.load(input_file)

YOLO_PERSON = 0

def process_image_resnet(image):
	try:
		img = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
		img = np.expand_dims(img, axis=0)
		img = tf.keras.applications.resnet50.preprocess_input(img)
	except Exception:
		print("********* Null image!!!")
		return
	return img
	
def drawBoundingBoxWithLabel(network_outputs):
	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	yolo_classIDs = []
	reid_classIDs = []
	reid_confidences = []
	label_name = ""
	
	# loop over each of the layer outputs
	for output in network_outputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			yolo_classID = np.argmax(scores)
			if yolo_classID == YOLO_PERSON:
				confidence = scores[yolo_classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > CONFIDENCE:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					
					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					crop_img = image[y : y + int(height), x : x + int(width)]
					if process_image_resnet(crop_img) is None:
						continue

					value = model.predict(process_image_resnet(crop_img))
					reid_classID = np.argmax(value)
					reid_confidence = value.max()

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					# yolo_classIDs.append(yolo_classID)
					reid_classIDs.append(reid_classID)
					reid_confidences.append(float(reid_confidence))

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE,
		THRESHOLD)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the image
			yolo_color = [int(c) for c in YOLO_COLORS[YOLO_PERSON]]
			reid_color = [int(c) for c in REID_COLORS[reid_classIDs[i]]]

			label = labels_dict[reid_classIDs[i]]
			# name = "Me" if label == "quang" else "Unknown"
			if label == "quang":
				reid_text = "ID: {} ({:4f})".format("Me", reid_confidences[i])

				cv2.rectangle(image, (x, y), (x + w, y + h), reid_color, 2)
				cv2.putText(image, reid_text, (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, reid_color, 2)

	# show the output image
	cv2.imshow("Image", image)
	# cv2.imshow('Image',  cv2.resize(image, (800, 600)))

if __name__ == "__main__":

	CONFIDENCE = 0.5
	THRESHOLD = 0.3

	YOLO_PATH = "yolo_model/"
	# IMAGE_FILE = "img/GQ_Style_Seoul04.webp"

	# load the COCO class labels our YOLO model was trained on
	labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	REID_CLASSES = model.layers[-1].output_shape[-1]

	# initialize a list of colors to represent each possible class label
	np.random.seed(0)

	REID_COLORS = np.random.randint(0, 255, size=(REID_CLASSES, 3),
		dtype="uint8")
	YOLO_COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	# derive the paths to the YOLO weights and model configuration
	weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
	configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])

	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


	# cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture("build_dataset/videos/testing/troll.mp4")
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	while cap.isOpened(): 
		# load our input image and grab its spatial dimensions
		# image = cv2.imread(IMAGE_FILE)
		
		ret, frame = cap.read()
		image = np.array(frame)
		(H, W) = image.shape[:2]

		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities
		# blobFromImage - Creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(ln)
		
		drawBoundingBoxWithLabel(layerOutputs)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			break
