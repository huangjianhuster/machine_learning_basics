import copy
import math
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

np.random.seed(0)

datapath = "./"

#################### Task 1 ###################
# Implement PCA
# You should be able to implement your own kmeans by using numpy only. 
def vector_draw(vec, ax, **kwargs):
    color = 'k'
    if 'color' in kwargs:
        color = kwargs['color']
    zorder = 3 
    if 'zorder' in kwargs:
        zorder = kwargs['zorder']
        
    veclen = math.sqrt(vec[0]**2 + vec[1]**2)
    head_length = 0.25
    head_width = 0.25
    vec = (veclen - head_length) / veclen * vec
    ax.arrow(0, 0, vec[0],vec[1], head_width=head_width, head_length=head_length, fc=color, ec=color, linewidth=3, zorder=zorder)
    

def compute_pcs(X, lam):
	P = float(X.shape[1])
	cov = 1 / P * np.dot(X, X.T) + lam * np.eye(X.shape[0])
	D, V = np.linalg.eigh(cov)
	return D, V


def pca_transform_data(X, **kwargs):
	num_components = X.shape[0]
	if 'num_components' in kwargs:
		num_components = kwargs['num_components']
          
	lam = 10 ** (-7)
	if 'lam' in kwargs:
		lam = kwargs['lam']
          
	D, V = compute_pcs(X, lam)
	V = V[:, -num_components:]
	D = D[-num_components:]

	W = np.dot(V.T, X)
	return W, V


def run_task1(): 
	# load in dataset
	csvname = datapath + '2d_span_data.csv'
	x = np.loadtxt(csvname, delimiter = ',')

	print(np.shape(x)) # (2, 50)

	# remove the mean from the data
	x = x - np.mean(x, axis=1, keepdims=True)

	# pca 
	W, V = pca_transform_data(x)
	
	# plot
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(121)
	ax.scatter(*x, color='k')
	vector_draw(V[:, 0], ax, **{'color': 'red'})
	vector_draw(V[:, 1], ax, **{'color': 'red'})
	ax.set_title('original space')
	ax.axis('equal')

	ax = fig.add_subplot(122)
	ax.scatter(*W, color='k')
	vector_draw(np.array([0, 1]), ax, **{'color': 'red'})
	vector_draw(np.array([1, 0]), ax, **{'color': 'red'})
	ax.set_title('PCA transformed space')
	ax.axis('equal')
	fig.tight_layout()
	plt.show()


#################### Task 2 ###################
# Implement K-Means; 
# You should be able to implement your own kmeans by using numpy only. 
def update_assignments(data, centroids):
    n_samples = data.shape[1]
    assignments = []
    for p in range(n_samples):
        # get p-th point
        x_p = data[:, p][:, np.newaxis]
        
        # compute distance between pth point and all centroids
        # using numpy broadcasting
        diffs = np.sum((x_p - centroids) ** 2, axis = 0)
        
        # determine closest centroid
        ind = np.argmin(diffs)
        assignments.append(ind)
    return np.array(assignments)


# update centroid locations
def update_centroids(data, old_centroids, assignments):
    K = old_centroids.shape[1]
    
	# new centroid container
    centroids = []
    for k in range(K):
        # collect indices of points belonging to kth cluster
        S_k = np.argwhere(assignments == k)
        
        # take average of points belonging to this cluster
        c_k = 0
        if np.size(S_k) > 0:
            c_k = np.mean(data[:, S_k],axis = 1)
        else:  # what if no points in the cluster?  keep previous centroid
            c_k = copy.deepcopy(old_centroids[:, k])[:, np.newaxis]
        centroids.append(c_k)
    centroids = np.array(centroids)[:, :, 0]
    return centroids.T


# main k-means function
def KMeans(data, n_clusters, max_its):
	# choose the centroids of the data
	indices = np.random.choice(data.shape[1], size=(n_clusters, ), replace=False)
	centroids = data[:, indices]
    
    # outer loop - alternate between updating assignments / centroids
	for j in range(max_its):
		# update cluter assignments
		assignments = update_assignments(data, centroids)
        
		# update centroid locations
		centroids = update_centroids(data, centroids, assignments)
        
	# final assignment update
	assignments = update_assignments(data, centroids)
	return centroids, assignments


def plot_clusters(data, centroids, assignments):
	fig = plt.figure(figsize=(7, 4))
	ax = fig.add_subplot(111)
	for i, cluster in enumerate(np.unique(assignments)):
		mask = assignments == cluster
		ax.scatter(*data[:, mask], color=sns.color_palette('deep')[i], label='cluster %d' % cluster)
		ax.scatter(*centroids[:, i], color=sns.color_palette('deep')[i], marker='x', s=100, label='centroid %d' % cluster)
	ax.legend()
	fig.tight_layout()
	plt.show()
     

# computer for the average error
def compuate_ave(data, centroids, assignments):
    P = len(assignments)
    K = centroids.shape[1]
    error = 0
    for k in range(K):
        centroid = centroids[:, k][:, np.newaxis]
        mask = assignments == k
        if np.sum(mask) > 0:
            error += np.sum(np.linalg.norm(data[:, mask] - centroid))
    
	# divide by the average
    error /= float(P)
    return error


def scree_plot(data, K_range, max_its):
    # initialize figure
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    
    ### outer loop - run K-means for each k ###
    K_errors = []
    for k in K_range:
        errors = []
        for _ in range(5):
            # run K-means algo
            centroids, assignments = KMeans(data, k, max_its)

            # compute average error over dataset
            error = compuate_ave(data, centroids, assignments)
            errors.append(error)
            
        # take final error
        best_ind = np.argmin(errors)
        K_errors.append(errors[best_ind])
    
    # plot cost function value for each K chosen    
    ax.plot(K_range,K_errors,'ko-')
    
    # dress up panel
    ax.set_xlabel('number of clusters')
    ax.set_ylabel('objective value')
    ax.set_xticks(K_range)
    fig.tight_layout()
    plt.show()
 

def run_task2(): 
	# Loading the data
	P = 50 # Number of data points
	data = datasets.make_blobs(n_samples = P, centers = 3, random_state = 10)[0]
	data = data.T
	print(data.shape) # (2, 50)
    
	# TODO: fill in your code 
	n_clusters = 3
	max_its = 100
	centroids, assignments = KMeans(data, n_clusters, max_its)
	plot_clusters(data, centroids, assignments)

	K_range = np.arange(1, 10+.001, 1, dtype=int)
	scree_plot(data, K_range, max_its)


if __name__ == '__main__':
	# run_task1()
	run_task2() 

