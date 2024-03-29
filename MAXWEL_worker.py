import numpy as np
import dill
from sklearn.model_selection import train_test_split
import time

from keras.layers import Dense, Lambda ,Input, LSTM, GRU, Dropout, Conv1D, Bidirectional, Flatten, Reshape, Permute, concatenate
from keras.models import Model, load_model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.hyperparameters import UniformFloatHyperparameter as UFH
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter as UIH

class MAXWEL_worker(Worker):
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		
		self.batch_size=16

		self.data_info=dill.load(open('Temp_Data/data_info.pkl','rb'))

		self.X_train=dill.load(open('Temp_Data/X_train.pkl','rb'))
		self.Y_train=dill.load(open('Temp_Data/Y_train.pkl','rb'))

		self.X_valid=dill.load(open('Temp_Data/X_valid.pkl','rb'))
		self.Y_valid=dill.load(open('Temp_Data/Y_valid.pkl','rb'))
		
		self.X_test=dill.load(open('Temp_Data/X_test.pkl','rb'))
		self.Y_test=dill.load(open('Temp_Data/Y_test.pkl','rb'))

		self.fit_cnt=self.data_info['fit_cnt']
	  
	def compute(self,config,budget,working_directory,*args,**kwargs):

		model=Sequential()

		#####################################################################
		# Input #############################################################
		#####################################################################

		# model.add(Input(input_shape = (self.data_info['train_time'],self.data_info['num_features'],)))

		#####################################################################
		# Attention Layer ###################################################
		#####################################################################
		
		model.add(Permute((2,1),input_shape = (self.data_info['train_time'],self.data_info['num_features'],)))
		model.add(GRU(config['num_temporal_GRU1'], recurrent_dropout=config['GRU_dropout_rate'],  return_sequences = True))
		model.add(Dropout(rate=config['dropout_layer_rate']))
		model.add(Conv1D(config['num_temporal_conv'], config['size_temporal_conv'], padding="valid"))
		model.add(GRU(config['num_temporal_GRU2'], recurrent_dropout=config['GRU_dropout_rate'], return_sequences=True))
		model.add(Flatten())	
		
		#####################################################################
		# Decision Layers ###################################################
		#####################################################################

		model.add(Dense(config['num_dense1'], activation="linear"))
		model.add(Dropout(rate=config['dropout_layer_rate']))
		model.add(Dense(config['num_dense2'], activation='linear'))
		model.add(Dense(self.data_info['predict_time'], activation="linear"))
				
		#####################################################################
		# Model Definition ##################################################
		#####################################################################
		
		model.compile(loss='mean_squared_error', optimizer=Adam(lr=config['learning_rate']))

		print('============================================================================')
		print('Name: {}'.format(self.data_info['primary']))
		# print('Station {} of {}'.format(self.data_info['station_cnt'],self.data_info['num_stations']))
		print('Fit: {}'.format(self.fit_cnt))
		print('============================================================================')

		self.fit_cnt+=1

		model.fit(
				self.X_train, 
				self.Y_train, 
				batch_size=self.batch_size, 
				epochs=int(budget),
				verbose=2)

		train_score=model.evaluate(self.X_train,self.Y_train, verbose=0)
		valid_score=model.evaluate(self.X_valid,self.Y_valid, verbose=0)

		Y_predict=model.predict(self.X_valid)
		Y_test=model.predict(self.X_test)

		def columnar_R2(mat_predict,mat_true):
			out=np.zeros(4)
			for i in range(4):
				out[i]=1-np.linalg.norm(mat_predict[:,i]-mat_true[:,i])/np.linalg.norm(mat_predict[:,i]-mat_true[:,i].mean())
			return out

		valid_R2=columnar_R2(Y_predict,self.Y_valid)
		test_R2=columnar_R2(Y_test,self.Y_test)

		self.data_info['fit_cnt']=self.fit_cnt

		dill.dump(self.data_info,open("Temp_Data/data_info.pkl",'wb'))

		np.set_printoptions(precision=2)

		print('============================================================================')
		print("Validation R2: {}".format(valid_R2))
		print("Hypervalidation R2: {}".format(test_R2))
		print('============================================================================')
		print()
		print()

		return({'loss': -sum(valid_R2)/4, 
				'info': {	
						'num_pars':model.count_params(), 
						'train':train_score, 
						'valid':valid_score,
						'valid_R2':list(valid_R2),
						'test_R2':list(test_R2)}})


	@staticmethod
	def get_configspace():
		cs=CS.ConfigurationSpace()
		learning_rate=UFH(		'learning_rate',
								lower=1e-6,upper=1e-1,
								default_value=1e-2,log=True)

		GRU_dropout_rate=UFH(	'GRU_dropout_rate',
								lower=.1,upper=.4,
								default_value=.3,log=True)

		dropout_layer_rate=UFH(	'dropout_layer_rate',
								lower=.1,upper=.4,
								default_value=.3,log=True)

		num_temporal_GRU1=UIH(	'num_temporal_GRU1',
								lower=10,upper=50,
								default_value=20,log=False)

		num_temporal_conv=UIH(	'num_temporal_conv',
								lower=10,upper=50,
								default_value=15,log=False)

		size_temporal_conv=UIH(	'size_temporal_conv',
								lower=2,upper=7,
								default_value=3,log=False)

		num_temporal_GRU2=UIH(	'num_temporal_GRU2',
								lower=10,upper=50,
								default_value=20,log=False)

		num_dense1=UIH(			'num_dense1',
								lower=10,upper=100,
								default_value=20,log=False)

		num_dense2=UIH(			'num_dense2',
								lower=10,upper=100,
								default_value=20,log=False)

		cs.add_hyperparameters([learning_rate,
								GRU_dropout_rate,
								dropout_layer_rate,
								num_temporal_GRU1,
								num_temporal_conv,
								size_temporal_conv,
								num_temporal_GRU2,
								num_dense1,
								num_dense2])
		return cs