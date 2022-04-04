#Written by Oliver AS Lyon
#University of Calgary
#2022

#LIBRARY IMPORTS
import csv							#IO 
import matplotlib.pyplot as plt		#graphs
import math							#used for log/exp
import numpy as np					#used for data structures etc...
from sklearn.mixture import GaussianMixture	#used for GMM
import scipy.stats as st			#stats lib

"""
Data_import:
	This function is used to read the data from the file.
	ADD MORE DOCS ON FUNCTIONALITY
"""
def Data_import(file_name,log_base):
	data = []
	with open("NHANES_DATA/" + file_name, newline='\n') as f:
		reader = csv.reader(f)
		for row in reader:
			if row[1] != '':
				data.append(float(row[1]))

	plt.hist(data,bins=30)
	plt.title('Data Hist')
	plt.show()

	#---------------------------------------------------------------------
	#LOG TRANSFORM THE DATA
	#---------------------------------------------------------------------

	log_data = []
	for i in data:
		log_data.append(math.log(i,log_base))
	log_data = np.array(log_data).reshape(-1,1)
	data = np.array(data).reshape(-1,1)

	#---------------------------------------------------------------------
	#VIS DIST OF DATA
	#---------------------------------------------------------------------

	plt.hist(log_data,bins=30)
	plt.title('Log Data Hist')
	plt.show()
	
	#---------------------------------------------------------------------
	#SUMMARY OF DATA
	#---------------------------------------------------------------------

	print(f"""
	NUM SAMPLES:{len(data)}
	NUM NON-DIA:{len([i for i in data if i < 5.7])}
	NUM PRE: {len([i for i in data if 5.7 <= i < 6.5])}
	NUM DIA: {len([i for i in data if 6.5 <= i])}""")
	return(log_data)
	
"""
Fmm:
	NOTES
"""
def Fmm(log_data):
	#FMM
	#---------------------------------------------------------------------
	N = 2 #NUMBER OF DISTRIBUTIONS

	gm = GaussianMixture(n_components=N).fit(log_data)

	fig = plt.figure(figsize=(9, 7))
	plt.rcParams['font.size'] = '20'
	ax = fig.add_subplot(111)

	x = np.linspace(0,6,1000)
	logprob= gm.score_samples(x.reshape(-1,1))
	resp = gm.predict(x.reshape(-1,1))
	pdf = np.exp(logprob)
	pdf_indi = resp * pdf[:, np.newaxis]

	#---------------------------------------------------------------------
	#PLOT HIST AND MODEL
	#---------------------------------------------------------------------
	plt.rcParams['font.size'] = '20'
	ax.hist(log_data,30,density=True,histtype='stepfilled',alpha=0.4)
	ax.plot(x, pdf, '-k')
	ax.set_xlabel('log(HbA1c%)')
	ax.set_ylabel('Frequency')
	#change font size
	#plt.savefig('HIST.png', dpi=600)
	
	plt.show()

	#---------------------------------------------------------------------
	#SAVE PARAMETERS
	#---------------------------------------------------------------------

	p1 = gm.weights_[0]
	mu1 = gm.means_[0,0]
	var1 = gm.covariances_[0,0,0]
	p2 = gm.weights_[1]
	mu2 = gm.means_[1,0]
	var2 = gm.covariances_[1,0,0]

	#---------------------------------------------------------------------
	#CDF PLOTS
	#---------------------------------------------------------------------

	x = np.arange(0,max(log_data),0.001)
	y1 = st.norm(mu1,np.sqrt(var1))
	y2 = st.norm(mu2,np.sqrt(var2))
	y1_vals = p1 * y1.cdf(x)
	y2_vals = p2 * y2.cdf(x)

	y = []
	#add the distrubtiutions together
	for i in range(0,len(x)):
		y.append(y1_vals[i]+y2_vals[i])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('log(HbA1c%)')
	ax.set_ylabel('Cumulative Frequency')

	plt.plot(x,y)

	#plot log data
	X_data = np.sort(list(log_data.T)[0])
	CDF_data = np.cumsum(len(X_data)*[1])/len(X_data)

	plt.plot(X_data,CDF_data)
	plt.show()

	#---------------------------------------------------------------------
	#PRINT PARAMETERS FROM MODEL
	#---------------------------------------------------------------------
	print(f"""
MODEL PARAMS
---------------------------------------------------------------------
Goodness of Fit: {gm.score(log_data)} 
% OF EACH DISTRIBUTION
%1:  {gm.weights_[0]}
%2: {gm.weights_[1]}
MEANS OF EACH DISTRIBUTION
MU1: {gm.means_[0,0]}
MU2: {gm.means_[1,0]}
VARIANCE OF EACH DISTRIBUTION
VAR1: {gm.covariances_[0,0,0]}
VAR2: {gm.covariances_[1,0,0]}""")
	
	return([p1,mu1,var1,p2,mu2,var2])
	
"""
Samples the population
"""
def Sampler(models,pop,log_base):
	p1   = models[0]
	mu1  = models[1]
	var1 = models[2]
	p2   = models[3]
	mu2  = models[4]
	var2 = models[5]

	#randomly sample from both distributions
	pop_cat = np.random.rand(pop)
	bins = [p2]
	cat = list(np.digitize(pop_cat,bins))
	pop1 = cat.count(1)
	pop2 = pop-pop1

	samples = list(np.random.normal(mu1, np.sqrt(var1), pop1))
	samples2 = list(np.random.normal(mu2, np.sqrt(var2), pop2))

	samples.extend(samples2)

	plt.hist(samples,300,density=True)
	plt.title("sample")
	plt.show()

	#remove log transofrmation
	data = np.power(log_base, samples)

	plt.hist(data,300,density=True)
	plt.title("Data for simultation")
	plt.show()
	
	return(data)
	
"""
write data
"""
def Write_pop(file_name, data):
	with open("SIM_POPULATION/"+file_name, 'w') as myfile:
		wr = csv.writer(myfile)
		wr.writerow(data)
	
	
	
	