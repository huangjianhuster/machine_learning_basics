import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

np.random.seed(0)

datapath = "./"

#################### Task 1 ###################

# Implement PCA
# You should be able to implement your own PCA by using numpy only. 

def run_task1(): 
	# load in dataset
	csvname = datapath + '2d_span_data.csv'
	x = np.loadtxt(csvname, delimiter = ',')

	print(np.shape(x)) # (2, 50)
	
    # TODO: fill in your code 


#################### Task 2 ###################

# Implement K-Means; 
# You should be able to implement your own kmeans by using numpy only. 

def run_task2(): 
	# Loading the data
	P = 50 # Number of data points
	blobs = datasets.make_blobs(n_samples = P, centers = 3, random_state = 10)
	data = np.transpose(blobs[0])
	print(data.shape) # (2, 50)
    
	# TODO: fill in your code 



if __name__ == '__main__':
	run_task1()
	run_task2() 