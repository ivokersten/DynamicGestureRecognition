#!/usr/bin/env python
import cv2
import numpy as np

import roslib
import rospy
import sys,time

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray


input_file = 'example.avi'
cap = cv2.VideoCapture(input_file)

class VideoClient:
	global cap
	
	def __init__(self):
		self.image_pub = rospy.Publisher("compressed_image", CompressedImage, queue_size=1)
		self.rate = rospy.Rate(20)
		self.rsp = rospy.Subscriber("gesture", Int32MultiArray, self.callback, queue_size=1)

	def callback(self, rospy_data):
		print(str(rospy_data.data))

	def SendVideo(self):
		while cap.isOpened():
			ret, frame = cap.read()
			if ret == True:

				frame = cv2.flip(frame, 1)

				msg = CompressedImage()
				msg.header.stamp = rospy.Time.now()
				msg.format = 'jpeg'
				msg.data = np.array(cv2.imencode('.jpeg', frame)[1]).tostring()
				self.image_pub.publish(msg)

				cv2.imshow('Transitted image', frame)
				#self.rate.sleep()
				if cv2.waitKey(50) & 0xFF == ord('q'):
					break
			else:
				cv2.destroyAllWindows()
				break


if __name__ == '__main__':
	rospy.init_node('Hand_detection_client', anonymous=True)
	vc = VideoClient()
	vc.SendVideo()
