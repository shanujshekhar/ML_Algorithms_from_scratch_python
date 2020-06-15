import argparse
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import preprocessing

def euclidian_distance(first, second):
	distance = 0

	for i in range(len(first)):
		distance += (first[i] - second[i])**2

	return np.sqrt(distance)

def manhattan_distance(first, second):
	distance = 0

	for i in range(len(first)):
		distance += np.abs((first[i] - second[i]))

	return distance

def square_rooted(vector):
	return np.round(np.sqrt(np.sum([a*a for a in vector])),3)

def cosine_similarity(first, second):
	numerator = np.sum(a*b for a,b in zip(first, second))
	denominator = square_rooted(first)*square_rooted(second)
	return (1 - numerator/float(denominator))

def initial_centroids(k, dataset):
	indices = np.random.randint(len(dataset), size=k)
	centroids = [dataset[i] for i in indices]
	return centroids

def assign_clusters(centroids, dataset, distance_metric):

	clusters = []

	for obs_no, observation in enumerate(dataset):
		dists = []
		
		for cluster, centroid in enumerate(centroids):

			if distance_metric=='cosine':
				dist = cosine_similarity(dataset[obs_no], centroid)
			elif distance_metric=='euclidian':
				dist = euclidian_distance(dataset[obs_no], centroid)
			elif distance_metric=='manhattan':
				dist = manhattan_distance(dataset[obs_no], centroid)

			if dist!=0:
				dists.append([cluster, dist])

		dists.sort(key = lambda x: x[1])
		
		clusters.append(dists[0][0])
		
	return clusters

def cal_new_centroids(k, clusters, dataset):

	clusters_mean = {}

	for i in range(k):
		clusters_mean[i] = []

	for cluster, data in zip(clusters, dataset):
		clusters_mean[cluster].append(data)

	centroids = [[]] * k
	sum_squares = []

	for cluster, data in clusters_mean.items():
		mean = np.mean(data, axis=0)
		centroids[cluster] = mean
		
		mean_repeated = []

		for i in range(len(data)):
			mean_repeated.append(mean)

		mean_repeated = np.array(mean_repeated)
		sum_squares.append(np.sum(np.sum((data - mean_repeated)**2)))


	return centroids, sum_squares

def evaluate(clusters, dataset):
	correct = 0

	for cluster_label, data in zip(clusters, dataset):
		if cluster_label == int(data[len(data)-1]):
			correct += 1

	return correct/len(data)

def convergence(new_clusters, old_clusters):

	dissimilar = 0

	for n, o in zip(new_clusters, old_clusters):

		if n!=o:
			dissimilar += 1

	if dissimilar == 0:
		return True, dissimilar
	else:
		return False, dissimilar

def plot(epochs, dissimilar):

	epochs = np.arange(epochs)
	plt.plot(epochs, dissimilar, marker = 'o', linestyle = ':')
	plt.title('Difference in Clusters vs Epochs(' + str(len(epochs)) + ')')
	plt.xlabel('Epochs')
	plt.ylabel('Difference in Clusters')
	plt.show()

def find_percent_labels(k, clusters, labels):

	cluster_describe = {}

	for i in range(k):
		cluster_describe[i] = {'diag_0' : 0, 'diag_1' : 0}

	diag_0, diag_1 = 0, 0

	for cluster, label in zip(clusters, labels):
		
		if int(label)==0:
			diag_0 += 1
			cluster_describe[cluster]['diag_0'] += 1
		else:
			diag_1 += 1
			cluster_describe[cluster]['diag_1'] += 1
		
	print ('Clusters Description')
	print ()
	for cluster, details in cluster_describe.items():
		totalPoints = details['diag_0'] + details['diag_1']
		print ('Cluster ', cluster)
		print ('Diagnosis 0 labels %: ', details['diag_0']/totalPoints * 100)
		print ('Diagnosis 1 labels %: ', details['diag_1']/totalPoints * 100)
		print ()


def main():

	parser = argparse.ArgumentParser(description='K Means Clustering')
	parser.add_argument('--dataset', type=str, help='path to dataset')
	parser.add_argument('--distance', type=str, help='distance metric', default='euclidian')

	args = parser.parse_args()

	# Number of clusters
	k = 2
	data = np.loadtxt(args.dataset, dtype=str, delimiter=',')
	
	labels = data[1:,-1]
	dataset = np.array(data[1:,0:len(data[0])-1]).astype(np.float)

	min_max_scaler = preprocessing.MinMaxScaler()
	dataset = min_max_scaler.fit_transform(dataset)
	
	cluster_vars = []
	centroids = initial_centroids(k, dataset)

	i = 0
	dissimilar = []

	print ('Distance Used: ', args.distance)

	while True:

		clusters = assign_clusters(centroids, dataset, args.distance)
		centroids, sum_squares = cal_new_centroids(k, clusters, dataset)

		cluster_vars.append(np.mean(sum_squares))

		if i==0:
			dissimilar.append(len(dataset))
		else:
			converge, dis = convergence(clusters, old_clusters)
			dissimilar.append(dis)
			if converge:
				print ('Converged at epoch: ', i+1)
				print ()
				break

		old_clusters = clusters

		i += 1

	find_percent_labels(k, clusters, labels)

	plot(i+1, dissimilar)

	

if __name__ == '__main__':
	main()