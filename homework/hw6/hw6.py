import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

np.random.seed(0)

datapath = "./"

#################### Task 1 ###################

# Implement PCA
# You should be able to implement your own PCA by using numpy only. 
def center(x):
    standardized_x = (x.T - x.mean(axis=1)).T
    return standardized_x

def compute_pcs(x, lam):
    p = float(x.shape[1])
    cov = 1/p * np.dot(x, x.T) + lam*np.eye(x.shape[0]) # adding lam to prevent numerical stability
    D, V = np.linalg.eigh(cov)
    return D, V

def pca_transform_data(x, **kwargs):
    num_components = x.shape[0]
    if 'num_components' in kwargs:
        num_components = kwargs["num_components"]
    lam = 1e-7
    if "lam" in kwargs:
        lam = kwargs["lam"]
    
    D, V = compute_pcs(x, lam)
    V = V[:, -num_components:]
    D = D[-num_components:]
    
    # compute transformed data for PC space
    w = np.dot(V.T, x)
    return w, V

def run_task1(): 
	# load in dataset
	csvname = datapath + '2d_span_data.csv'
	x = np.loadtxt(csvname, delimiter = ',')

	print(np.shape(x)) # (2, 50)
	
    # TODO: fill in your code 
	standardized_x = center(x)
	kwargs = {'num_components':2, 'lam':1e-7}
	w, V = pca_transform_data(standardized_x, **kwargs)

	# plot
	fig, axs = plt.subplots(1,2,figsize=(8,5), gridspec_kw={'width_ratios':[2,1]})
	axs[0].plot(standardized_x[0], standardized_x[1], "bo")
	axs[0].set_xticks([-5,-2.5,0,2.5,5])
	axs[0].set_xlim([-6, 6])
	axs[0].set_xlabel("x1", fontsize=14)
	axs[0].set_ylabel("x2", fontsize=14)
	axs[0].tick_params(axis='both', which='major', labelsize=14)
	axs[0].axhline(y = 0, color = 'k', linestyle = '--')
	axs[0].axvline(x = 0, color = 'k', linestyle = '--')
	axs[0].arrow(0, 0, V[0,0], V[0,1], color='r', width=0.1)
	axs[0].arrow(0, 0, V[1,0], V[1,1], color='r', width=0.1)
	axs[0].set_title("Original data")

	axs[1].plot(w[0], w[1], "go")
	axs[1].set_xticks([-2.5,0,2.5])
	axs[1].set_xlim([-2.7, 2.7])
	axs[1].set_xlabel("c1", fontsize=14)
	axs[1].set_ylabel("c2", fontsize=14)
	axs[1].tick_params(axis='both', which='major', labelsize=14)
	axs[1].axhline(y = 0, color = 'k', linestyle = '--')
	axs[1].axvline(x = 0, color = 'k', linestyle = '--')
	axs[1].arrow(0, 0, 0, 1, color='r', width=0.1)
	axs[1].arrow(0, 0, 1, 0, color='r', width=0.1)
	axs[1].set_title("Encoded data")

	plt.tight_layout()
	plt.savefig("task1.png")
	plt.show()


#################### Task 2 ###################

# Implement K-Means; 
# You should be able to implement your own kmeans by using numpy only. 

def k_means(data, iterations, centroid_num):
    x1 = np.random.rand(1, centroid_num)
    x2 = np.random.rand(1, centroid_num)
    centroids_x1 = x1*( data[0].max() - data[0].min()  ) + data[0].min()
    centroids_x2 = x2*( data[1].max() - data[1].min()  ) + data[1].min()
    centroids = np.vstack([centroids_x1, centroids_x2])
    centroids_history = [centroids, ]
    # calculate average distance as the cost?
    cost_history = []
    for i in range(1, iterations+1):
        # determine classes
        # use numpy broadcasting
        # see: https://stackoverflow.com/questions/68185016/numpy-broadcasting-two-arrays-of-different-shape-to-matrix
        distances = np.linalg.norm(data.T - centroids.T.reshape(centroid_num,1,2), axis=2)
        
        index = distances.argmin(axis=0)
        
        new_cens = []
        dist_to_cens = 0
        for cen in range(centroid_num):
            tmp = data[:,index==cen]

            if tmp.size != 0:
                # print(type(tmp))
                new_cens.append(tmp.mean(axis=1))
            else: # maintain the same as previous ones
                new_cens.append(centroids[:,cen])
            # if empty?
            dist_to_cens += distances[:, index==cen][cen].sum()
        cost = dist_to_cens/data.shape[1]
        cost_history.append(cost)
        # update centroids
        centroids = np.array(new_cens).T
        centroids_history.append(centroids)
    return centroids_history, cost_history

def run_task2(): 
	# Loading the data
	P = 50 # Number of data points
	blobs = datasets.make_blobs(n_samples = P, centers = 3, random_state = 10)
	data = np.transpose(blobs[0])
	print(data.shape) # (2, 50)
    
	# TODO: fill in your code

	# plot original data
	plt.plot(data[0], data[1], "ko")
	plt.savefig("task2-orginal-data.png")
	plt.tight_layout()
	plt.show()

	# k-means centroid number = 3
	cost_tmp = 100	# a large number, doesn't matter...
	for iter in range(5):
		cen_hist, cost_hist = k_means(data, 30, 3)
		if cost_hist[-1] < cost_tmp:
			cost_tmp = cost_hist[-1]
			cen_final = cen_hist[-1]
	distances = np.linalg.norm(data.T - cen_final.T.reshape(3,1,2), axis=2)
	index = distances.argmin(axis=0)
	
	print("cen_final", cen_final.T)
	
	# print(index)
	color_dict = {0: 'orange', 1: 'blue', 2: 'green'}
	fig, ax = plt.subplots()
	for g,cen in zip(np.unique(index), cen_final.T):
		print(g, cen)
		ix = np.where(index == g)
		ax.scatter(data[0][ix], data[1][ix], c = color_dict[g], label = g, s = 50)
		ax.scatter(cen[0], cen[1], s=100, c=color_dict[g], marker='*', label=f"Centroid-{g}", linewidth=0.5, edgecolor='black')
	ax.tick_params(axis='both', which='major', labelsize=14)
	plt.legend(frameon=True)
	plt.tight_layout()
	plt.savefig("task2-cennum3.png")
	plt.show()

	# k-means for centroid number = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
	final_cens = {}
	for cen_num in range(1,11,1):
		# run each for 6 random trials <-- k-means method is sensitive to initialization
		cost_tmp = 100	# a large number, doesn't matter...
		for iter in range(10):
			cen_hist, cost_hist = k_means(data, 30, cen_num)
			if cost_hist[-1] < cost_tmp:
				cost_tmp = cost_hist[-1]
				cen_tmp = cen_hist[-1]
		final_cens[cen_num] = cen_tmp, cost_tmp
	# print(final_cens)

	# plot
	cost_arr = [i[1][1] for i in final_cens.items()]
	plt.plot(np.arange(1,11,1), cost_arr, "go-")
	plt.xlabel("number of clusters", fontsize=14)
	plt.xticks(np.arange(1,11,1))
	plt.ylabel("cost", fontsize=14)
	plt.tick_params(axis='both', which='major', labelsize=14)
	plt.tight_layout()
	plt.savefig("task2-cluster-numbers.png")
	plt.show()

	# for myself
	# plot every figure with different numbers of clusters
	fig, ax = plt.subplots(2, 5, figsize=(20,10))
	for idx,i in enumerate(ax.flatten()):
		i.plot(data[0], data[1], "ko", markersize=2.5)
		i.set_title(f"cluster={idx+1}")
		i.tick_params(axis='both', which='major', labelsize=6)
	for axi, (cluster_num, centroid) in zip(ax.flatten(), final_cens.items()):
		# print(cluster_num, centroid)
		axi.plot(centroid[0][0], centroid[0][1], "r*" )
	plt.show()




if __name__ == '__main__':
	# run_task1()
	run_task2() 