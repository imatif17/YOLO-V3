import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from utils import *
from darknet import Darknet


# Set the location and name of the cfg file
cfg_file = './cfg/yolov3.cfg'

# Set the location and name of the pre-trained weights file
weight_file = './weights/yolov3.weights'

# Set the location and name of the COCO object classes file
namesfile = 'data/coco.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)
nms_thresh = 0.6
iou_thresh = 0.4


cam = cv2.VideoCapture(0)
while(True):
	ret,frame = cam.read()
	if(ret):
		frame = cv2.flip(frame,1)
		resized_image = cv2.resize(frame, (m.width, m.height))
		boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
		#print_objects(boxes, class_names)

		img = plot_boxes(frame, boxes, class_names, plot_labels = True)
		cv2.imshow('YOLO v3',img)
		
		key = cv2.waitKey(1) & 0xFF
		if(key == ord('q')):
			break
	else:
		break

cam.realease()
cv2.destroyAllWindows()
