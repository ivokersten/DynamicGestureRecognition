from keras.models import model_from_json
import numpy as np
import math
import os
import csv
import cv2
from utils import detector_utils as detector_utils

PATH_TO_TRAIN_DATA_FILE = 'Motion_Classification/data.csv'

PATH_TO_NN = 'NN_files/'

def initialize_datafile():
    if not os.path.exists(PATH_TO_TRAIN_DATA_FILE):
        with open(PATH_TO_TRAIN_DATA_FILE, mode='w') as data_file:
            datawriter = csv.writer(data_file, delimiter=',')

            i = 0
            row = []
            for index in range(15):
                row.append('x'+str(index))
                row.append('y'+str(index))
                i += 2
            row.extend(['horizontal','vertical','clockwise circle', 'counterclockwise circle', 'nothing'])
            print(row)
            datawriter.writerow(row)
            print('Created data file')

def write_path_to_datafile(lastPos, cap_params, meanx, meany, added_pos, pred, PREDICT_GESTURE):
    if added_pos < 2:
        gesture_row = [0,0,0,0,0]
        i = added_pos
        row = []
        for coord in range(len(lastPos[i])):
            for pos in range(len(lastPos[i][coord])):
                if pos == 0:
                    row.append(int(lastPos[i][coord][pos]*cap_params['im_width'])-meanx[i])
                else:
                    row.append(int(lastPos[i][coord][pos]*cap_params['im_height'])-meany[i])
        if PREDICT_GESTURE:
            gesture_row[pred[i]] = 1
            row.extend(gesture_row)
        else:
            row.extend([0,0,0,0,1])
        with open(PATH_TO_TRAIN_DATA_FILE, mode='a') as data_file:
            datawriter = csv.writer(data_file, delimiter=',')
            datawriter.writerow(row)
    else:
        for i in range(0,2):
            gesture_row = [0,0,0,0,0]
            row = []
            for coord in range(len(lastPos[i])):
                for pos in range(len(lastPos[i][coord])):
                    if pos == 0:
                        row.append(int(lastPos[i][coord][pos]*cap_params['im_width'])-meanx[i])
                    else:
                        row.append(int(lastPos[i][coord][pos]*cap_params['im_height'])-meany[i])
            if PREDICT_GESTURE:
                gesture_row[pred[i]] = 1
                row.extend(gesture_row)
            else:
                row.extend([0,0,0,0,1])
            with open(PATH_TO_TRAIN_DATA_FILE, mode='a') as data_file:
                datawriter = csv.writer(data_file, delimiter=',')
                datawriter.writerow(row)

def load_net():
    json_file = open(PATH_TO_NN + 'model.json','r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights(PATH_TO_NN + 'model.h5')
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def initialize():
    lastPos = [[],[]]
    lastOutput = [[],[]]
    for i in range(15):
        lastPos[0].append((1,0))
        lastPos[1].append((0,0))
    for i in range(15):
        for j in range(2):
            lastOutput[j].append(4)
    return lastPos, lastOutput

def find_hands(boxes, scores, cap_params):
    positions = []
    count = 0
    #Select the most probable hand positions
    for i in range(cap_params['num_hands_detect']):
        if (scores[i] > cap_params['score_thresh']):
            positions.extend(((boxes[count][1]+boxes[count][3])/2,(boxes[count][0]+boxes[count][2])/2))
            count += 1
    return positions

def update_lastpos(positions, lastPos):
    added_pos = 2
    #Append new hand position to the correct list
    if len(positions) == 2:
        #If the detected hand is closer to the first list, add it there, and to the second list otherwise
        if (math.sqrt((lastPos[0][0][0]-positions[0])**2+(lastPos[0][0][1]-positions[1])**2) < math.sqrt((lastPos[1][0][0]-positions[0])**2+(lastPos[1][0][1]-positions[1])**2)):
            #print(str(math.sqrt((lastPos[0][0][0]-positions[0])**2+(lastPos[0][0][1]-positions[1])**2)) + " < " + str(math.sqrt((lastPos[1][0][0]-positions[0])**2+(lastPos[1][0][1]-positions[1])**2)))
            lastPos[0] = lastPos[0][-1:] + lastPos[0][:-1]
            lastPos[0][0] = positions[0:2]
            added_pos = 0
        else:
            #print(str(math.sqrt((lastPos[0][0][0]-positions[0])**2+(lastPos[0][0][1]-positions[1])**2)) + " > " + str(math.sqrt((lastPos[1][0][0]-positions[0])**2+(lastPos[1][0][1]-positions[1])**2)))
            lastPos[1] = lastPos[1][-1:] + lastPos[1][:-1]
            lastPos[1][0] = positions[0:2]
            added_pos = 1
    else:
        #Calculate distances between new hand positions and previous additions to lists
        distlist = [0,0,0,0]
        for hand in range(0,2):
            for listnum in range (0,2):
                distlist[2*hand + listnum] = math.sqrt((lastPos[listnum][0][0]-positions[hand*2])**2+(lastPos[listnum][0][1]-positions[hand*2+1])**2)

        #print(str(distlist))
        minindex = np.argmin(distlist)
        if minindex == 0 or minindex == 3:
            lastPos[0] = lastPos[0][-1:] + lastPos[0][:-1]
            lastPos[0][0] = positions[0:2]
            lastPos[1] = lastPos[1][-1:] + lastPos[1][:-1]
            lastPos[1][0] = positions[2:]
        else:
            lastPos[0] = lastPos[0][-1:] + lastPos[0][:-1]
            lastPos[0][0] = positions[2:]
            lastPos[1] = lastPos[1][-1:] + lastPos[1][:-1]
            lastPos[1][0] = positions[0:2]


        '''
        #If first position is closer to first list, add it there
        if (math.sqrt((lastPos[0][0][0]-positions[0])**2+(lastPos[0][0][1]-positions[1])**2) < math.sqrt((lastPos[0][0][0]-positions[2])**2+(lastPos[0][0][1]-positions[3])**2)):
            lastPos[0] = lastPos[0][-1:] + lastPos[0][:-1]
            lastPos[0][0] = positions[0:2]

            lastPos[1] = lastPos[1][-1:] + lastPos[1][:-1]
            lastPos[1][0] = positions[2:]
        else:
            lastPos[0] = lastPos[0][-1:] + lastPos[0][:-1]
            lastPos[0][0] = positions[2:]
            lastPos[1] = lastPos[1][-1:] + lastPos[1][:-1]
            lastPos[1][0] = positions[0:2]
        '''
    return lastPos, added_pos

def predict_gestures(lastPos,cap_params,model,lastOutput, added_pos):
    meanx,meany = [0,0],[0,0]
    predicted_gestures = [0,0]
    predicted_gestures_raw = [0,0]
    #Draw paths described by hands
    for hand in range(len(lastPos)):
        tempx, tempy = 0, 0
        for i in range(len(lastPos[hand])):
            tempx += lastPos[hand][i][0]*cap_params['im_width']
            tempy += lastPos[hand][i][1]*cap_params['im_height']
        meanx[hand] = int(tempx/len(lastPos[hand]))
        meany[hand] = int(tempy/len(lastPos[hand]))

        #print(lastPos)
        posarray = []
        for coord in range(len(lastPos[hand])):
            for pos in range(len(lastPos[hand][coord])):
                if pos == 0:
                    posarray.append(int(lastPos[hand][coord][pos]*cap_params['im_width'])-meanx[hand])
                else:
                    posarray.append(int(lastPos[hand][coord][pos]*cap_params['im_height'])-meany[hand])
        
        posarray = np.array([posarray,])
        pred = np.argmax(model.predict(posarray))

        if added_pos == hand or added_pos == 2:
            lastOutput[hand] = lastOutput[hand][-1:] + lastOutput[hand][:-1]
            lastOutput[hand][0] = pred
            predicted_gestures_raw[hand] = pred
       
        #pred = max(set(lastOutput[hand]), key = lastOutput[hand].count)

        if lastOutput[hand].count(lastOutput[hand][0]) == len(lastOutput[hand]): #If all 15 last predicted gestures are equal, this is the prediction, else no gesture
            #print("All gestures equal in list " + str(hand))
            #print("Hand " + str(hand) + ": " + str(lastOutput[hand].count(lastOutput[hand][0])) + " , " + str(len(lastOutput[hand])))
            pred = lastOutput[hand][0]
        else:
            pred = 4

        predicted_gestures[hand] = pred

    #print(lastOutput)
    return predicted_gestures, lastOutput, meanx, meany, predicted_gestures_raw


def construct_output(image_np,cap_params,lastPos,prediction, scores, boxes):
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    #Draw boxes on image
    detector_utils.draw_box_on_image(cap_params['num_hands_detect'], cap_params['score_thresh'], scores, boxes, cap_params['im_width'], cap_params['im_height'], image_np)
    #Draw path on image
    for hand in range(len(lastPos)):
        for i in range(len(lastPos[hand])):
            if( i < len(lastPos[hand])-1):
                if(hand == 0):
                    cv2.line(image_np, (int(lastPos[hand][i][0]*cap_params['im_width']),int(lastPos[hand][i][1]*cap_params['im_height'])), (int(lastPos[hand][i+1][0]*cap_params['im_width']),int(lastPos[hand][i+1][1]*cap_params['im_height'])), (255,0,0),2)
                else:
                    cv2.line(image_np, (int(lastPos[hand][i][0]*cap_params['im_width']),int(lastPos[hand][i][1]*cap_params['im_height'])), (int(lastPos[hand][i+1][0]*cap_params['im_width']),int(lastPos[hand][i+1][1]*cap_params['im_height'])), (0,255,0),2)
    #Draw detected gestures on image
    for hand in range(2):
        if hand == 0:
            if prediction[hand] == 0:
                cv2.putText(image_np, "Horizontal", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif prediction[hand] == 1:
                cv2.putText(image_np, "Vertical", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif prediction[hand] == 3:
                cv2.putText(image_np, "Counter clk circle", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif prediction[hand] == 2:
                cv2.putText(image_np, "Clk circle", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif prediction[hand] == 4:
                cv2.putText(image_np, "No gesture", (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        else:
            if prediction[hand] == 0:
                cv2.putText(image_np, "Horizontal", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif prediction[hand] == 1:
                cv2.putText(image_np, "Vertical", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif prediction[hand] == 3:
                cv2.putText(image_np, "Counter clk circle", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif prediction[hand] == 2:
                cv2.putText(image_np, "Clk circle", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif prediction[hand] == 4:
                cv2.putText(image_np, "No gesture", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image_np
    