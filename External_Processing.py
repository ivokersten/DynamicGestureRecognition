#!/usr/bin/env python
import cv2
import numpy as np
from utils import detector_utils as detector_utils
from utils import gesture_utils as gesture_utils
import datetime
from multiprocessing import Queue
import threading

import roslib
import rospy
import sys, time, os

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray

import csv
import tensorflow as tf

#Display processed frames for debugging
DISPLAY_OUTPUT = True
#Write recorded hand paths to datafile for testing
RECORD_PATHS = False
#Write recorded paths to datafile with the gesture predicted by the network
PREDICT_GESTURE = False
#Track processing speed
TRACK_FPS = False
#Save output frames to video file
SAVE_OUTPUT = False

NUM_THREADS = 4

INPUT_QUEUE_SIZE = 1

output_path = 'output.avi'

input_lock = threading.Lock()
input_cv = threading.Condition(input_lock)
frame_lock = threading.Lock()
frame_cv = threading.Condition(frame_lock)
next_frame = 0 #Tracks the current frame number to reconstruct output in original order
TERMINATE = False

#Worker calls all functions to obtain the estimated gesture
def Assembler(worker_q, worker_frame_q, cap_params):
    global DISPLAY_OUTPUT
    global RECORD_PATHS
    global PREDICT_GESTURE
    global TERMINATE
    global SAVE_OUTPUT, output_path
    global next_frame
    print("Assembler started!")
    prediction = [4,4] #Initialize predition
    
    prev_prediction = [4, 4]

    #Load network to recognize gestures
    model = gesture_utils.load_net(os.path.abspath(os.getcwd()))
    #Initialize lastPos and lastOutput with default values
    lastPos, lastOutput = gesture_utils.initialize()
    start_time = None
    frameCounter = 0 #To count elapsed frame
    idleTimer = time.time()
    PROCESSING = False

    processing_stats = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] #(Total time, total frames, fps), frames with 0 hands, frames with 1 hand, frames with 2 hands

    publisher = rospy.Publisher("gesture", Int32MultiArray, queue_size=1)

    if SAVE_OUTPUT:
        out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'),24,(int(cap_params['im_width']),int(cap_params['im_height'])))

    while not (PROCESSING and (time.time() - idleTimer) > 10):  #After starting, the Assembler exits after not receiving new info for 10 seconds
        if not worker_q.empty():
            PROCESSING = True #Becomes true when at least one frame has been received
            idleTimer = time.time()

            #print("Assembler acquired frame")
            if start_time == None and TRACK_FPS: #Note time the first frame is processed
                start_time = time.time()

            #Obtain access to queue of worker output
            frame_cv.acquire()
            if DISPLAY_OUTPUT or SAVE_OUTPUT:
                image_np = worker_frame_q.get()
            worker_output = worker_q.get()
            next_frame += 1
            frame_cv.notifyAll()
            frame_cv.release()

            boxes = worker_output[0]
            scores = worker_output[1]

            positions = gesture_utils.find_hands(boxes, scores, cap_params) #Obtain center positions of max 2 most probable hands

            if(len(positions) > 0): #If at least one hand is detected
                lastPos, added_pos = gesture_utils.update_lastpos(positions,lastPos) #Append new position to closest previous position to keep hands separate
                prediction, lastOutput, meanx, meany, predicted_gestures_raw = gesture_utils.predict_gestures(lastPos, cap_params,model, lastOutput, added_pos) #Obtain predicted gesture based on new input
                print('Prediction: ' + str(prediction))

                if prev_prediction[0] != prediction[0]:
                    print(prediction[0])

                    response = Int32MultiArray()
                    response.data = [prediction[0]]
                    publisher.publish(response)

                    prev_prediction[0] = prediction[0]

                if prev_prediction[1] != prediction[1]:
                    
                    response = Int32MultiArray()
                    response.data = [prediction[1]]
                    publisher.publish(response)

                    prev_prediction[1] = prediction[1]

                if RECORD_PATHS: #If paths need to be recorded to CSV file
                    gesture_utils.write_path_to_datafile(lastPos, cap_params, meanx, meany, added_pos, predicted_gestures_raw, PREDICT_GESTURE)
            
            if TRACK_FPS: #Track overall FPS as well as on specific cases
                frame_time = time.time()-start_time
                processing_stats[0][0] += frame_time
                processing_stats[0][1] += 1

                if len(positions) == 0:
                    processing_stats[1][0] += frame_time
                    processing_stats[1][1] += 1
                elif len(positions) == 2:
                    processing_stats[2][0] += frame_time
                    processing_stats[2][1] += 1
                else:
                    processing_stats[3][0] += frame_time
                    processing_stats[3][1] += 1

                start_time = time.time()

            if DISPLAY_OUTPUT or SAVE_OUTPUT: #If output frame should be created
                image_np = gesture_utils.construct_output(image_np, cap_params, lastPos, prediction, scores, boxes) #Draw output frame with additional data to screen

                if SAVE_OUTPUT:
                    out.write(image_np)

                if DISPLAY_OUTPUT:
                    cv2.imshow('detected hands', image_np)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        TERMINATE = True
                        pass
            
            if processing_stats[0][1] - frameCounter == 10 and TRACK_FPS: #Update FPS and display
                for i in range(0, len(processing_stats)): 
                    if processing_stats[i][0] > 0:
                        processing_stats[i][2] = processing_stats[i][1] / processing_stats[i][0]
                print(processing_stats)
                frameCounter = processing_stats[0][1]

    print('Assembler exiting')
    TERMINATE = True
    try:
        input_cv.notifyAll()
    except:
        pass
    
    if SAVE_OUTPUT:
        out.release()

def Worker(input_q, frameNumber_q, worker_q, worker_frame_q, id):
    global TERMINATE
    idleTimer = time.time()  #Saves time at start of last frame
    PROCESSING = False #Indicates a video stream is being processed
    msg_timer = time.time()
    #Load hand recognition graph
    detection_graph, sess = detector_utils.load_inference_graph()

    while not (PROCESSING and (time.time() - idleTimer) > 10):
        if not input_q.empty():
            PROCESSING = True #Becomes true when at least one frame has been received
            idleTimer = time.time()

            input_cv.acquire()
            while input_q.empty():
                if TERMINATE:
                    input_cv.release()
                    break
                input_cv.wait()
            if TERMINATE:
                break
            image_np = input_q.get()
            frameNumber = frameNumber_q.get()
            input_cv.notifyAll()
            input_cv.release()

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) #Convert frame to correct color format
            
            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess) #Detect hands
            

            frame_cv.acquire()

            while frameNumber != next_frame:
                frame_cv.wait()
            worker_q.put([boxes, scores])

            if DISPLAY_OUTPUT or SAVE_OUTPUT:
                worker_frame_q.put(image_np)
            frame_cv.release()

    print("Worker "+str(id)+" exited")


class FrameProcessor:
    frameNumber = 0
    def __init__(self, input_q, frameNumber_q):
        #self.detection_graph, self.sess = detector_utils.load_inference_graph()
        self.receiver = rospy.Subscriber("compressed_image", CompressedImage, self.callback, queue_size=1)
        
        self.input_q = input_q
        self.frameNumber_q = frameNumber_q
        self.output_q = output_q
        self.framenumber = 0

    def callback(self, ros_data):
        #Prepare incoming frame
        img = np.fromstring(ros_data.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        input_cv.acquire()
        while input_q.qsize() >= INPUT_QUEUE_SIZE:
            input_cv.wait()
        input_q.put(img)
        frameNumber_q.put(self.frameNumber)
        input_cv.notifyAll()
        input_cv.release()
        self.frameNumber += 1
        

if __name__ == '__main__':
    if RECORD_PATHS:
        gesture_utils.initialize_datafile() #Initialize datafile if it does not exist yet

    #Initialize queues
    input_q = Queue(maxsize=INPUT_QUEUE_SIZE)
    frameNumber_q = Queue(maxsize=INPUT_QUEUE_SIZE)
    worker_q = Queue(maxsize=2)
    worker_frame_q = Queue(maxsize=2)
    output_q = Queue(maxsize=2)

    #Create callback for when a frame comes in
    fp = FrameProcessor(input_q, frameNumber_q)
    rospy.init_node('Hand_detection_processing', anonymous=True)
     
    #Initialize parameters
    cap_params = {}
    cap_params['im_width'] = 320
    cap_params['im_height'] = 180
    cap_params['score_thresh'] = 0.2
    cap_params['num_hands_detect'] = 2


    #Create and start worker thread(s)
    worker_threads = []
    for i in range(0,NUM_THREADS):
        worker_threads.append(threading.Thread(target=Worker, args=(input_q, frameNumber_q, worker_q, worker_frame_q, i)))
        worker_threads[i].start()

    #Run assembler
    Assembler(worker_q, worker_frame_q, cap_params)

    #Clean up worker threads
    for i in range(0, len(worker_threads)):
        worker_threads[i].join()

    cv2.destroyAllWindows()
    sys.exit()