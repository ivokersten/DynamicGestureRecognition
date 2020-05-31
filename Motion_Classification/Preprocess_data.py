import pandas as pd 
import numpy as np
import cv2

raw_data_path = 'data.csv'
MAX_SAMPLES = 1000

org_data = pd.read_csv(raw_data_path)
org_data.dropna(axis = 0, how='any', thresh=None, subset=None, inplace = True)
org_data = org_data.astype(int)

#Sort input data by gesture category to ease correction of errors
gesture_data = [org_data.loc[org_data['horizontal'] == 1], org_data.loc[org_data['vertical'] == 1], org_data.loc[org_data['clockwise circle'] == 1], org_data.loc[org_data['counterclockwise circle'] == 1], org_data.loc[org_data['nothing'] == 1]]

for i in range(0,len(gesture_data)):
	gesture_data[i] = gesture_data[i][:].copy()
	print(len(gesture_data[i].index))

data = pd.concat(gesture_data).reset_index(drop=True)


print('horizontal: ' + str(data.loc[:,'horizontal'].sum()))
print('vertical: ' + str(data.loc[:,'vertical'].sum()))
print('counterclockwise circle: ' + str(data.loc[:,'counterclockwise circle'].sum()))
print('clockwise circle: ' + str(data.loc[:,'clockwise circle'].sum()))
print('empty: ' + str(data.loc[:,'nothing'].sum()))

index = 0

#for index,row in data.iterrows():
while index < len(data.index) and index >= 0:
	row = data.loc[index,:]
	x = row[:30:2].tolist()
	y = row[1:30:2].tolist()
	img = np.zeros((400,600,3))
	for i in range(len(x)-2): #Draw gesture from green to red
		cv2.line(img,((x[i]+150)*2,(y[i]+100)*2), ((x[i+1]+150)*2,(y[i+1]+100)*2), (0,i*1/10, max([1-i*1/10,0])), 2)
	if row[30] == 1:
		cv2.putText(img, "Horizontal", (100*2, 20*2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
	elif row[31] == 1:
		cv2.putText(img, "Vertical", (100*2, 20*2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)	
	elif row[32] == 1:
		cv2.putText(img, "Clockwise Circle", (100*2, 20*2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)	
	elif row[33] == 1:
		cv2.putText(img, "CounterClockwise Circle", (100*2, 20*2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
	else:
		cv2.putText(img, "Nothing", (100*2, 20*2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

	cv2.putText(img, str(index), (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
	cv2.imshow('shape',img)

	hold = True
	while hold:
		key = cv2.waitKey(1) & 0xFF
		if(key == ord('n') or key == ord('d') or key == ord('1') or key == ord('2') or key == ord('3') or key == ord('4') or key == ord('5') or key == ord('q') or key == ord('p') or key == ord('s')):
			hold = False
	if key == ord('d'):  #d to remove frame from data file
		data = data.drop(index)
	elif key == ord('p'): #p to step back one frame, n to continue to next frame
		index -= 2
	elif key == ord('s'):
		img = np.clip(img * 255, 0, 255) # proper [0..255] range
		img = img.astype(np.uint8)  # safe conversion
		cv2.imwrite('Data_Collection.jpg',img)
	elif key == ord('1'):
		data.loc[index,'horizontal'] = 1
		data.loc[index,'vertical'] = 0
		data.loc[index,'counterclockwise circle'] = 0
		data.loc[index,'clockwise circle'] = 0
		data.loc[index,'nothing'] = 0
		print("horizontal registered")
	elif key == ord('2'):
		data.loc[index,'horizontal'] = 0
		data.loc[index,'vertical'] = 1
		data.loc[index,'counterclockwise circle'] = 0
		data.loc[index,'clockwise circle'] = 0
		data.loc[index,'nothing'] = 0
	elif key == ord('3'):
		data.loc[index,'horizontal'] = 0
		data.loc[index,'vertical'] = 0
		data.loc[index,'counterclockwise circle'] = 1
		data.loc[index,'clockwise circle'] = 0
		data.loc[index,'nothing'] = 0
	elif key == ord('4'):
		data.loc[index,'horizontal'] = 0
		data.loc[index,'vertical'] = 0
		data.loc[index,'counterclockwise circle'] = 0
		data.loc[index,'clockwise circle'] = 1
		data.loc[index,'nothing'] = 0
	elif key == ord('5'):
		data.loc[index,'horizontal'] = 0
		data.loc[index,'vertical'] = 0
		data.loc[index,'counterclockwise circle'] = 0
		data.loc[index,'clockwise circle'] = 0
		data.loc[index,'nothing'] = 1
	elif key == ord('q'): #q to quit and save changes
		break
	index += 1

print(data.describe())
cv2.destroyAllWindows()


gesture_data = [data.loc[data['horizontal'] == 1], data.loc[data['vertical'] == 1], data.loc[data['clockwise circle'] == 1], data.loc[data['counterclockwise circle'] == 1], data.loc[data['nothing'] == 1]]

#Shuffle data to ensure no whole gestures are cut from the dataset

shuffled_data = data.sample(frac=1).reset_index(drop=True).copy()
trimmed_gesture_data = [shuffled_data.loc[shuffled_data['horizontal'] == 1], shuffled_data.loc[shuffled_data['vertical'] == 1], shuffled_data.loc[shuffled_data['clockwise circle'] == 1], shuffled_data.loc[shuffled_data['counterclockwise circle'] == 1], shuffled_data.loc[shuffled_data['nothing'] == 1]]


#Write preprocessed data to .csv, one version with MAX_SAMPLES of each gesture and one with the complete dataset

for i in range(0,len(trimmed_gesture_data)):
	trimmed_gesture_data[i] = trimmed_gesture_data[i][:MAX_SAMPLES].copy()
	gesture_data[i] = gesture_data[i][:].copy()

trimmed = pd.concat(trimmed_gesture_data)
data = pd.concat(gesture_data)

trimmed.to_csv('trimmed_data.csv', index=False, header=True)
data.to_csv('data.csv', index=False, header=True)