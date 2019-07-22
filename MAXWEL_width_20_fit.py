def MAXWEL_bandster_station(raw_name,width):

	name=raw_name.split('.')
	if name[-1]!="csv":
		return

	name=name[0]
	df=pd.read_csv('Train_Data/Width_'+width+'/'+name+'.csv').drop(columns=['time'])

	try:
		result_dir='Results/'+name
		os.mkdir(result_dir)
	except:
		result_dir='Results/'+name

	assert df.isna().sum().sum()==0, name+" has NANs"

	data_info={}
	data_info['primary']=name
	data_info['num_features']=df.shape[1]
	data_info['train_time']=12
	data_info['predict_time']=4
	data_info['num_samples']=df.shape[0]

	X=np.zeros((data_info['num_samples'], data_info['train_time'], df.shape[1]))
	Y=np.zeros((data_info['num_samples'], data_info['predict_time']))
	for i in range(data_info['num_samples']-(data_info['train_time']+data_info['predict_time']+1)):
		X[i,:,:]=df.iloc[i:i+data_info['train_time']].values
		Y[i,:]=np.asarray(df[name+'_scaled_demand'][i+data_info['train_time']:i+data_info['train_time']+data_info['predict_time']].values)

	idx=sample(range(len(X)),9000)
	train_idx=idx[:6000]
	valid_idx=idx[6000:8000]
	test_idx=idx[8000:]

	data_info['num_train_samples']=len(train_idx)
	data_info['train_idx']=train_idx

	data_info['num_valid_samples']=len(valid_idx)
	data_info['valid_idx']=valid_idx

	data_info['num_test_samples']=len(test_idx)
	data_info['test_idx']=test_idx

	data_info['fit_cnt']=1

	# data_info['station_cnt']=station_cnt
	# data_info['num_stations']=len(names)

	dill.dump(data_info,open("Temp_Data/data_info.pkl",'wb'))

	X_train=np.zeros(
		(data_info['num_train_samples'],
		 data_info['train_time'],
		 data_info['num_features']))
	Y_train=np.zeros(
		(data_info['num_train_samples'],
		 data_info['predict_time']))

	X_valid=np.zeros(
		(data_info['num_valid_samples'],
		 data_info['train_time'],
		 data_info['num_features']))
	Y_valid=np.zeros(
		(data_info['num_valid_samples'],
		 data_info['predict_time']))

	X_test=np.zeros(
		(data_info['num_test_samples'],
		 data_info['train_time'],
		 data_info['num_features']))
	Y_test=np.zeros(
		(data_info['num_test_samples'],
		 data_info['predict_time']))

	for i,ind in enumerate(data_info['train_idx']):
		X_train[i,:,:]=X[ind,:,:]
		Y_train[i,:]=Y[ind,:]

	for i,ind in enumerate(data_info['valid_idx']):
		X_valid[i,:,:]=X[ind,:,:]
		Y_valid[i,:]=Y[ind,:]

	for i,ind in enumerate(data_info['test_idx']):
		X_test[i,:,:]=X[ind,:,:]
		Y_test[i,:]=Y[ind,:]

	dill.dump(X_train,open("Temp_Data/X_train.pkl",'wb'))
	dill.dump(Y_train,open("Temp_Data/Y_train.pkl",'wb'))

	dill.dump(X_valid,open("Temp_Data/X_valid.pkl",'wb'))
	dill.dump(Y_valid,open("Temp_Data/Y_valid.pkl",'wb'))

	dill.dump(X_test,open("Temp_Data/X_test.pkl",'wb'))
	dill.dump(Y_test,open("Temp_Data/Y_test.pkl",'wb'))

	# Import a worker class
	from MAXWEL_worker import MAXWEL_worker as worker

	#Build an argument parser       
	parser = argparse.ArgumentParser(description='MAXWEL - sequential execution.')
	parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=5)
	parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=20)
	parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=10)
	parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=1)
	parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')

	args=parser.parse_args()

	#Define a realtime result logger
	result_logger = hpres.json_result_logger(directory=result_dir, overwrite=True)


	#Start a nameserver
	NS = hpns.NameServer(run_id='MAXWEL', host='127.0.0.1', port=None)
	NS.start()

	#Start the workers
	workers=[]
	for i in range(args.n_workers):
		w = worker(nameserver='127.0.0.1',run_id='MAXWEL', id=i)
		w.run(background=True)
		workers.append(w)

	#Define and run an optimizer
	bohb = BOHB(configspace = w.get_configspace(),
				run_id = 'MAXWEL',
				result_logger=result_logger,
				min_budget=args.min_budget, 
				max_budget=args.max_budget) 

	res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)

	#Shutdown the nameserver
	bohb.shutdown(shutdown_workers=True)
	NS.shutdown()




import pandas as pd
import numpy as np
import dill

import os
import argparse
import logging
import time

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from random import sample



#Toggle the following to un/mute the nameserver chatter
# logging.basicConfig(level=logging.WARNING)

# Warm start option
warm_start=False

#Check for previous runs, set pointer to latest, make new results directory
# run_cnt=-1
# file_list=os.listdir()
# for file in file_list:
# 	temp=set(file.split('_'))
# 	if {'WTTcast','run'}.issubset(temp):
# 		run_cnt+=1

# os.mkdir('WTTcast_run_'+str(run_cnt+1))

# Data preprocessing
names=os.listdir('Train_Data/Width_20')

for name in names:
	MAXWEL_bandster_station(name,'20')