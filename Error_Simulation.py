import numpy as np
import scipy.stats as st
import csv
import random
from multiprocessing import Pool
import multiprocessing
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import os

"""

"""
def Perterbation(input):
	CV, Bias, pre_data, cut_offs = input
	index = list(range(0,len(pre_data)))
	pre_bin = np.digitize(pre_data,cut_offs,right=False)
	rands = np.asarray(st.norm.rvs(size=(len(pre_data))))
	post_data = [(pre_data[i] + (rands[i] * CV * pre_data[i]) + (Bias * pre_data[i])) for i in index]
	post_bin = np.digitize(post_data, cut_offs, right=False)
 
	#for each catagory find sens/spec
	mat = metrics.confusion_matrix(pre_bin,post_bin)
	TP = []
	FP = []
	TN = []
	FN = []
	r = []

	for i in range(0,len(mat[0])):
		#true Postives
		TP = (float(mat[i,i]))
		#True Negatives
		temp = 0
		all_other_groups = list(range(0,len(mat[0])))
		all_other_groups.remove(i)
		for j in all_other_groups:
			for k in all_other_groups:
				temp += mat[k,j]
		TN = (float(temp))

		#False Positives
		FP = (float(sum(mat[0:i,i])+sum(mat[i+1:,i])))
		
		r.append((CV, Bias, TP/float(sum(mat[i,:])),TN/ (TN+FP)))
        
	return(r)
	
"""

"""
def Error_simulation(x,cut_offs, bias_min, bias_max, cv_min, cv_max, step_size):
	
	#Introducing error
	pool = Pool(processes=multiprocessing.cpu_count())
	results = list()
	tasks = list()
	inputs = list() 
	for CV in np.arange(cv_min,(cv_max+step_size),step_size):
		for Bias in np.arange(bias_min,(bias_max + step_size), step_size):  
			inputs.append((CV,Bias,x,cut_offs))
			
	for result in tqdm.tqdm(pool.imap_unordered(Perterbation,inputs), total=len(inputs)):	
		results.append(result)
	return(results)

"""

"""	
def Save_simulation(file_name,results,cut_offs):
	for i in range(0,len(cut_offs)+1):
		name = 'SIMULATION_DATA/' + str(i) + '_' + file_name
		try:
			file = open(name ,'a')
			file.close()
		except:
			raise Error("Unexpected error in new data creation.")
		
		#prep for writing
		a = csv.writer(open(name, 'w', newline=''))
		a.writerow(['Group' + str(i)])
		a.writerow(['CV','BIAS','SENSITIVITY','SPECIFICTIY'])

		for result in results:
			a.writerow(result[i])

"""

"""			
def Plotter(file):
	#reading in the data
	path = os.path.join("SIMULATION_DATA/"+file)
	rows = list(csv.reader(open(path),delimiter=','))

	rows.pop(0)#removing tiles
	rows.pop(0)#removing tiles

	CV = list()
	BIAS = list()
	one = list()
	two = list()

	#how many decimals to round too
	rond = 2

	maxCV = 0
	minCV = 0
	maxBIAS = 0 
	minBIAS = 0

	#Splitting the data
	for data in rows:
		 CV.append(round(float(data[0]),rond)*100)
		 BIAS.append(round(float(data[1]),rond)*100)
		 one.append(float(data[2]))
		 two.append(float(data[3]))
		 if round(float(data[0]),rond) < minCV:
			 minCV = round(float(data[0]),rond)
		 if round(float(data[0]),rond) > minCV:
			 maxCV = round(float(data[0]),rond)
		 if round(float(data[1]),rond) < minBIAS:
			 minBIAS = round(float(data[1]),rond)
		 if round(float(data[1]),rond) > minBIAS:
			 maxBIAS = round(float(data[0]),rond)

	x = sorted(set(CV)) 
	y = sorted(set(BIAS)) 
	sens = np.empty([len(y),len(x)])
	spec = np.empty([len(y),len(x)])


	for k in range(0,len(x)):
		for m in range(0,len(y)):
			#search for the correct CV and BIAS value
			for i in range(0,len(rows)):
				if(CV[i] == x[k]) and (BIAS[i] == y[m]):
					sens[m][k] = (one[i])*100

	for k in range(0,len(x)):
		for m in range(0,len(y)):
			#search for the correct CV and BIAS value
			for i in range(0,len(rows)):
				if(CV[i] == x[k]) and (BIAS[i] == y[m]):
					spec[m][k] = (two[i])*100
	plt.ioff()
	plt.figure(figsize=(13, 13))

	#add ATE lines
	ATE1 = 3
	ATE2 = 6
	ATE3 = 10

	plt.plot([0,ATE1/1.65],[ATE1,0],color='black',linestyle='--')
	plt.plot([0,ATE1/1.65],[-ATE1,0],color='black',linestyle='--')
	plt.plot([0,ATE2/1.65],[ATE2,0],color='black')
	plt.plot([0,ATE2/1.65],[-ATE2,0],color='black')
	plt.plot([0,ATE3/1.65],[ATE3,0],color='black',linestyle='--')
	plt.plot([0,ATE3/1.65],[-ATE3,0],color='black',linestyle='--')
	
	#the percent levels
	levels = [10,20,30,40,50,60,70,80,90]
	string = ['10','20','30','40','50','60','70','80','90']

	#change font size
	plt.rcParams['font.size'] = '20'
	
	CS1 = plt.contour(x,y,sens,levels, colors='red',linestyles='dashdot')
	CS2 = plt.contour(x,y,spec,levels, colors='blue')

	fmt1 = {}
	for l, s in zip(CS1.levels, string):
		fmt1[l] = s
	fmt2 = {}
	for l, s in zip(CS2.levels, string):
		fmt2[l] = s

	plt.xlabel("% Coefficient of Variation")
	plt.ylabel("% Bias")

	labels1 = plt.clabel(CS1,CS1.levels, inline=True, fmt=fmt1, fontsize=20, rightside_up=True, manual=False)
	labels2 = plt.clabel(CS2,CS2.levels, inline=True, fmt=fmt2, fontsize=20, rightside_up=True, manual=False)

	#upright numbers
	for l in labels1 + labels2:
		l.set_rotation(0)
	
	#add ATE lines
	ATE1 = 3
	ATE2 = 6
	ATE3 = 10

	plt.plot([0,ATE1/1.65],[ATE1,0],color='black',linestyle='--')
	plt.plot([0,ATE1/1.65],[-ATE1,0],color='black',linestyle='--')
	plt.plot([0,ATE2/1.65],[ATE2,0],color='black')
	plt.plot([0,ATE2/1.65],[-ATE2,0],color='black')
	plt.plot([0,ATE3/1.65],[ATE3,0],color='black',linestyle='--')
	plt.plot([0,ATE3/1.65],[-ATE3,0],color='black',linestyle='--')
	
	#plt.savefig('FIGURES' + file + '.png', dpi=600)
	plt.show()

			

       