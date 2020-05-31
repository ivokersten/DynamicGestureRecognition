from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import SGD
from keras import backend as K
import pandas as pd
import numpy as np
import csv
import gc
import seaborn as sns
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt

#Wether to create new sets of training and testing data or use previously created ones
CREATE_SETS = False
#Wether to compare the newly trained model to an already existing model. If false will overwrite a model when already present
COMPARE_TO_CURRENT_MODEL = True
#Wether to show accuracy and loss plots after training a model
SHOW_PLOTS = False

#Path to full set of preprocessed training and validation data
datapath = "trimmed_data.csv"

LEARNING_RATE = 0.0025
MOMENTUM = 0.85
ITERATIONS = 200

if CREATE_SETS:
	#Load data file
	data = pd.read_csv(datapath)
	data = data.astype(int)

	#Split data in 80% training and 20%  testing, ensuring equal amounts of each gesture are in the sets
	training_data = [data.loc[data['horizontal'] == 1], data.loc[data['vertical'] == 1], data.loc[data['clockwise circle'] == 1], data.loc[data['counterclockwise circle'] == 1], data.loc[data['nothing'] == 1]]
	testing_data = [data.loc[data['horizontal'] == 1], data.loc[data['vertical'] == 1], data.loc[data['clockwise circle'] == 1], data.loc[data['counterclockwise circle'] == 1], data.loc[data['nothing'] == 1]]

	for i in range(0,len(training_data)):
		training_data[i] = training_data[i][:int(len(training_data[i].index)*0.8)].copy()
		testing_data[i] = testing_data[i][int(len(testing_data[i].index)*0.8):].copy()
		print(len(training_data[i].index))
		print(len(testing_data[i].index))

	train = pd.concat(training_data).reset_index(drop=True)
	test = pd.concat(testing_data).reset_index(drop=True)

	#Shuffle order of data
	train = train.sample(frac=1).reset_index(drop=True)
	test = test.sample(frac=1).reset_index(drop=True)

	xtrain = train.loc[:, 'x0':'y14'].to_numpy()
	xtest = test.loc[:, 'x0':'y14'].to_numpy()
	ytrain = train.loc[:, 'horizontal':'nothing'].to_numpy()
	ytest = test.loc[:, 'horizontal':'nothing'].to_numpy()

	np.savetxt('testx.csv' ,xtest, delimiter=",")
	np.savetxt('testy.csv', ytest, delimiter=",")
	np.savetxt('trainx.csv', xtrain, delimiter=",")
	np.savetxt('trainy.csv', ytrain, delimiter=",")
else:
	try:
		xtest = np.loadtxt('testx.csv', dtype=int, delimiter=",")
		ytest = np.loadtxt('testy.csv', dtype=int, delimiter=",")
		xtrain = np.loadtxt('trainx.csv', dtype=int, delimiter=",")
		ytrain = np.loadtxt('trainy.csv', dtype=int, delimiter=",")
	except:
		print("\nNo predefined training and validation sets.\nRun script with 'CREATE_SETS = True' first")
		exit()
in_dim, out_dim = xtrain.shape[1], ytrain.shape[1]

if COMPARE_TO_CURRENT_MODEL:
	#Load model
	try:
		json_file = open('model.json','r')
	except:
		print("\nNo trained model.\nRun script with 'COMPARE_TO_CURRENT_MODEL = False' first")
	model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(model_json)
	loaded_model.load_weights('model.h5')
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	old_score = loaded_model.evaluate(xtest, ytest, verbose=0)[1]
	print("Score to beat: {}%\n\n".format(old_score*100))

	#Confution Matrix and Classification Report
	Y_pred = loaded_model.predict(xtest)
	y_pred = np.argmax(Y_pred, axis=1)
	print('Confusion Matrix')
	Matrix = confusion_matrix(y_true=np.argmax(ytest,axis=1), y_pred=y_pred)
	print(Matrix)

	class_names = ['Horizontal', 'Vertical', 'Clockwise Circle', 'Counterclockwise Circle', 'No Gesture']
	
	print(classification_report(y_true=np.argmax(ytest,axis=1), y_pred=y_pred, target_names=class_names))

	df_cm = pd.DataFrame(Matrix, index=class_names, columns=class_names)
	fig = plt.figure(figsize=(5,5))
	ax = fig.add_subplot(1,1,1)
	ax.figure.subplots_adjust(bottom=0.3, left=0.3)
	heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, square=True, annot_kws={"size": 7})
	heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=7)
	heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=7)
	plt.ylabel('True label', fontweight='bold')
	plt.xlabel('Predicted label', fontweight='bold')
	plt.savefig('Confusion_matrix.pdf')
	
#Set up early stopping callback
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

for model_try in range(ITERATIONS):
	#Create new model
	print("Creating model {} with learning rate {} and momentum {}".format(model_try, LEARNING_RATE, MOMENTUM))
	model = Sequential()
	model.add(Dense(150, input_dim=in_dim, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(75, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(out_dim, activation='softmax'))
	sgd = SGD(lr=LEARNING_RATE, momentum=MOMENTUM)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	#Fit model to trainin data
	history = model.fit(xtrain, ytrain, epochs=500, verbose=0, validation_data=[xtest,ytest], callbacks=[es])
	
	if SHOW_PLOTS:
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

	#Evaluate model performance
	model.predict(xtest)
	score = model.evaluate(xtest, ytest, verbose = 0)[1]
	print("Accuracy on test data: {}%".format(score*100))
	
	if not COMPARE_TO_CURRENT_MODEL:
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")
	else:
		#If new model performs better, overwrite saved model
		if(score > old_score):
			model_json = model.to_json()
			with open("model.json", "w") as json_file:
				json_file.write(model_json)
			model.save_weights("model.h5")
			#loaded_model = model

			print("model overwritten -- new accuracy: {}%. Original accuracy: {}%".format(score*100,old_score*100))
			old_score = score
	#Clean up model from memory to prevent memory leak		
	del model
	gc.collect()
	K.clear_session()
	tf.compat.v1.reset_default_graph()
