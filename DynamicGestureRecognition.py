#!/usr/bin/env python
import tensorflow as tf
import cv2
import numpy as np
from utils import detector_utils as detector_utils
from utils import gesture_utils as gesture_utils
import datetime
from multiprocessing import Queue, Pool
import threading

import sys, time

import csv

#Display processed frames for debugging
DISPLAY_OUTPUT = False
#Write recorded hand paths to datafile for testing
RECORD_PATHS = False
#Track processing speed
TRACK_FPS = True
#Save output frames to video file
SAVE_OUTPUT = False
output_path = '/home/ros/output.avi'

input_path = '/home/ros/example.avi'

#Worker calls all functions to obtain the estimated gesture
def worker(input_q,output_q,cap_params):
    global DISPLAY_OUTPUT
    global RECORD_PATHS
    global TERMINATE
    global SAVE_OUTPUT, output_path
    print("Worker started!")
    prediction = [4,4] #Initialize predition
    #Load hand recognition graph
    detection_graph, sess = detector_utils.load_inference_graph()
    #Load network to recognize gestures
    model = gesture_utils.load_net()
    #Initialize lastPos and lastOutput with default values
    lastPos, lastOutput = gesture_utils.initialize()
    start_time = None
    frameCounter = 0 #To count elapsed frames
    idleCounter = 0  #Counts iterations in which no new frames are available
    PROCESSING = False #Indicates a video stream is being processed

    processing_stats = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]] #(Total time, total frames, fps), frames with 0 hands, frames with 1 hand, frames with 2 hands

    if SAVE_OUTPUT:
        out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'),24,(int(cap_params['im_width']),int(cap_params['im_height'])))

    while idleCounter < 1000:
        if not input_q.empty():
            PROCESSING = True #Becomes true when at least one frame has been received
            idleCounter = 0

            if start_time == None and TRACK_FPS: #Note time the first frame is processed
                start_time = time.time()

            image_np = input_q.get()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) #Convert frame to correct color format

            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess) #Detect hands

            positions = gesture_utils.find_hands(boxes, scores, cap_params) #Obtain center positions of max 2 most probable hands

            if(len(positions) > 0): #If at least one hand is detected
                lastPos, added_pos = gesture_utils.update_lastpos(positions,lastPos) #Append new position to closest previous position to keep hands separate
                prediction, lastOutput, meanx, meany = gesture_utils.predict_gestures(lastPos, cap_params,model, lastOutput)
                
                if RECORD_PATHS: #If paths need to be recorded to CSV file
                    gesture_utils.write_path_to_datafile(lastPos, cap_params, meanx, meany, added_pos)
            
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
                        pass

            #Put response message in queue
            #output_q.put(prediction)
            print(prediction)
            
            if processing_stats[0][1] - frameCounter == 10 and TRACK_FPS: #Update FPS and display
                for i in range(0, len(processing_stats)): 
                    if processing_stats[i][0] > 0:
                        processing_stats[i][2] = processing_stats[i][1] / processing_stats[i][0]
                print(processing_stats)
                frameCounter = processing_stats[0][1]

        elif PROCESSING:
            idleCounter += 1

    print('Worker exiting')
    
    if SAVE_OUTPUT:
        out.release()


def video_processor(path,input_q):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 1)

                input_q.put(frame)
                
                #cv2.imshow('Transitted image', frame)

                #if cv2.waitKey(50) & 0xFF == ord('q'):
                #    break
                time.sleep(0.020)

            else:
                break
    cap.release()
    print("Video Processor done")

if __name__ == '__main__':
    if RECORD_PATHS:
        gesture_utils.initialize_datafile() #Initialize datafile if it does not exist yet

    input_q = Queue(maxsize=400)
    output_q = Queue(maxsize=4)
     
    cap_params = {}
    cap_params['im_width'] = 320
    cap_params['im_height'] = 180
    cap_params['score_thresh'] = 0.2
    cap_params['num_hands_detect'] = 2

    #Start worker as seperate thread
    #pool = Pool(1, worker, (input_q, output_q, cap_params))

    videoProcessing_thread = threading.Thread(target=video_processor, args=(input_path,input_q))
    videoProcessing_thread.start()

    try:
        worker(input_q, output_q, cap_params)
        videoProcessing_thread.join()
    except KeyboardInterrupt:
        print('Received KeyboardInterrupt')
        #pool.terminate()
        cv2.destroyAllWindows()
        sys.exit()