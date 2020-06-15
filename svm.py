import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

max_feature_value = 0
min_feature_value = 0

def generate_dataset():
	X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)

	# add one to the x-values to incorporate bias
	X1 = np.c_[np.ones((X0.shape[0])), X0]

	y = [-1 if label==0 else label for label in y]

	train_dict = {}
	test_dict = {}	

	train_dict['data'], train_dict['labels'] = X1[:int(0.8 * len(X0))], y[:int(0.8 * len(X0))]
	test_dict['data'], test_dict['labels'] = X1[int(0.8 * len(X0)):], y[int(0.8 * len(X0)):]
	
	return train_dict, test_dict

def train(data_dict):

	global max_feature_value
	global min_feature_value	

	theta = np.zeros(len(data_dict['data'][0]))

	max_feature_value = np.amax(data_dict['data'])
	min_feature_value = np.amin(data_dict['data'])

	weights_arr = []
	max_epochs = 100
	step_size = max_feature_value / 10

	np.random.seed(274)

	for epoch in range(1, max_epochs):
		
		weights = step_size * theta
		weights_arr.append(weights)

		index = np.random.randint(0, len(data_dict['data']))
		instance = data_dict['data'][index]
		label = data_dict['labels'][index]

		if( label * np.dot(weights, instance) < 1 ):
			theta = theta + (label * instance)			
			step_size /= 10


	weights = np.mean(weights_arr, axis=0)
	weights[0] = 1
	return weights	
	
def test(weights, test_dict):

	correct = 0

	for i, instance in enumerate(test_dict['data']):
		predicted_label = np.sign(np.dot(instance, weights))

		if( test_dict['labels'][i] == predicted_label ):
			correct += 1
	
	print ('Accuracy: ', correct/len(test_dict['data']) * 100)
	
def separatingLine(x, w, v):
	return (v - w[1] * x - w[0]) / w[2]

def draw(weights, data_dict):

	global max_feature_value
	global min_feature_value
    
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)  

	colors = ['b' if label==-1 else 'r' for label in data_dict['labels']]
	ax.scatter(data_dict['data'][:,1:2], data_dict['data'][:,2:], s = 20, facecolors='none', edgecolors=colors, linewidth=2)

	x_min = 0
	x_max = max_feature_value

	y1 = separatingLine(x_min, weights, -1.1)
	y2 = separatingLine(x_max, weights, -1.1)
	ax.plot([x_min, x_max], [y1, y2], 'y--', c='b')

	y1 = separatingLine(x_min, weights, -0.05)
	y2 = separatingLine(x_max, weights, -0.05)
	ax.plot([x_min, x_max], [y1, y2], 'k')
	

	y1 = separatingLine(x_min, weights, 1)
	y2 = separatingLine(x_max, weights, 1)
	ax.plot([x_min, x_max], [y1, y2], 'y--', c='r')

	plt.show()


def main():
	
	train_dict, test_dict = generate_dataset()
	weights = train(train_dict)
	
	test(weights, test_dict)
	draw(weights, train_dict)

if __name__ == "__main__":
	main()