import argparse
import numpy as np
import argparse
from sklearn import preprocessing

def split(data, test_size=0.25):
	test_indices = np.random.randint(len(data), size=int(len(data) * test_size))

	train_instances, test_instances, train_labels, test_labels = [], [], [], []

	train, test = {}, {}

	for i, row in enumerate(data):
		
		if i in test_indices:
			test_instances.append(row[0:len(row)-1])
			test_labels.append(row[-1])
		else:
			train_instances.append(row[0:len(row)-1])
			train_labels.append(row[-1])

	train['instances'], train['labels'] = train_instances, train_labels
	test['instances'], test['labels'] = test_instances, test_labels

	return train, test

def euclidian_distance(first, second):

	distance = 0

	for i in range(len(first)):
		distance += (first[i] - second[i])**2

	return np.sqrt(distance)

def predict(k, train, test_instance):

	dists = []

	for i in range(len(train['instances'])):
		dists.append([i, euclidian_distance(train['instances'][i], test_instance)])

	dists.sort(key = lambda x: x[1])

	labels_0, labels_1 = 0, 0

	for d in dists[:k]:
		if int(train['labels'][d[0]]) == 0:
			labels_0 += 1
		else:
			labels_1 += 1

	if labels_0 > labels_1:
		return 0
	else:
		return 1

def evaluate(k, train, test):

	correct = 0

	for i, test_instance in enumerate(test['instances']):
		predicted_label = predict(k, train, test_instance)

		if predicted_label == int(test['labels'][i]):
			correct += 1

	return correct/len(test['labels'])
	
def main():
	parser = argparse.ArgumentParser(description='K Nearest Neighbours')
	parser.add_argument('--dataset', type=str, help='path to dataset')
	parser.add_argument('--num-epochs', type=int, help='number of epochs')
	parser.add_argument('--k', type=int, help='number of neighbours')

	args = parser.parse_args()

	data = np.loadtxt(args.dataset, dtype=str, delimiter=',')
	dataset = np.array(data[1:]).astype(np.float)

	min_max_scaler = preprocessing.MinMaxScaler()
	dataset = min_max_scaler.fit_transform(dataset)

	accuracies = []
	for epoch in range(args.num_epochs):
		train, test = split(dataset, test_size=0.20)
		accuracy = evaluate(args.k, train, test)
		print ('Accuracy at epoch ', epoch+1, ' : ', accuracy)
		accuracies.append(accuracy)

	print ()
	print ('Average Accuracy over', args.num_epochs, 'epochs: ', np.mean(accuracies))


if __name__ == '__main__':
	main()