import argparse
import numpy as np

def kfold(instances, labels, folds):
	
	indices = np.arange(len(instances))
	firstFolds = len(instances)%folds
	foldsize = len(instances)//folds
	train_indices = []

	for _ in range(firstFolds):
		fold = []
		for _ in range(foldsize + 1):
			while True :
				val = np.random.choice(indices)
				if not any(val in sublist for sublist in train_indices) :
					break
			
			fold.append(val)
		train_indices.append(fold)
	for _ in range(folds - firstFolds):
		fold = []
		for _ in range(foldsize):
			while True :
				val = np.random.choice(indices)

				if not any(val in sublist for sublist in train_indices) :
					break
			fold.append(val)
		train_indices.append(fold)



	inputs = [[instances[index] for index in sublist] for sublist in train_indices]
	outputs = [[labels[index] for index in sublist] for sublist in train_indices]

	return inputs, outputs

def splitDataset(data):

	columns = data[0]

	labels = []
	ins = []

	count = 0
	for row in data:
		if count!=0:
			ins.append(row[0:(len(row)-1)])
			labels.append(row[-1])
		else:
			count += 1

	# Convert each instance from string to float
	instances = []
	for i in ins:
		instances.append(list(map(float, i)))

	labels = list(map(int, labels))
	
	return instances, labels

def ERM(instances, labels, epochs, learning_rate):

	weights = np.random.rand(len(instances[0]))
	bias = np.zeros(len(labels))
	prev_loss = None
	count = 0
	epoch = None

	for e in range(epochs):
		
		print ('Epoch No: ', e+1)
		loss = 0
		row = 0

		for train_inputs, label in zip(instances, labels):
			predicted_label = predict(train_inputs, weights, bias[row])
			weights += learning_rate * (label - predicted_label) * np.asarray(train_inputs)

			bias[row] += learning_rate * (label - predicted_label)

			loss += lossFunc(label, predicted_label)

			row += 1
		
		erm_loss = loss / len(instances)
		# print ('erm_loss: ', erm_loss)

		if prev_loss==erm_loss and (epoch==None or e==epoch+1):
			epoch = e
			count += 1
			
			if count==3:
				print ('Algorithm Terminated (Training Error has become constant for 4 epochs)')
				print ('Algorithm Terminated at Epoch: ',e+1)
				break
		else:
			count = 0
			epoch = None

		prev_loss = erm_loss

		if erm_loss==0:
			print ('Algorithm Terminated since it got converged (Training Error = 0)')
			print ('Algorithm Terminated at Epoch: ',e+1)
			break
	print ()
	print ('erm_loss: ', erm_loss)
	print ('weights: ', weights)
	print ('Accuracy: ', 1 - erm_loss)

def crossValidation(instances, labels, epochs, learning_rate):

	weights = np.random.rand(len(instances[0]))
	bias = np.zeros(len(labels))
	folds = 10
	folderror = 0

	inputs, outputs = kfold(instances, labels, folds)
	

	for i in range(folds):
		test_inputs = inputs[i]
		test_labels = outputs[i]
		train_inputs = []
		train_labels = []
		
		for j in range(folds):
			if j!=i:
				train_inputs.extend(inputs[j])
				train_labels.extend(outputs[j])

		for e in range(epochs):
		
			# print ('Epoch No: ', e+1)
			row = 0

			for inp, label in zip(train_inputs, train_labels):
				predicted_label = predict(inp, weights, bias[row])
				weights += learning_rate * (label - predicted_label) * np.asarray(inp)

				bias[row] += learning_rate * (label - predicted_label)
				row += 1

		row = 0
		loss = 0
		for inp, label in zip(test_inputs, test_labels):
			predicted_label = predict(inp, weights, bias[row])
			loss += lossFunc(label, predicted_label)

			row += 1
		
		print ('Fold Error: ', loss/len(test_inputs))
		folderror += loss/len(test_inputs)
	
	print ('Mean Fold Error: ', folderror/folds)
	print ('weights: ', weights)
	print ('Accuracy: ', 1 - (folderror/folds))			
		
def lossFunc(label, predicted_label):

	# 0/1 Loss
	if label==predicted_label:
		return 0
	else:
		return 1

def predict(train_inputs, weights, bias):
	epoch_output = np.dot(train_inputs, weights) + bias

	# Activation Function
	if epoch_output > 0 :
		return 1
	else:
		return 0


def main():
	parser = argparse.ArgumentParser(description='Perceptron Implementation')
	parser.add_argument('--dataset', type=str, help='dataset location')
	parser.add_argument('--mode', type=str, help='erm or kfold')
	parser.add_argument('--num-epochs', type=int, help='number of epochs')

	args = parser.parse_args()

	learning_rate = 0.001

	data = np.loadtxt(args.dataset, dtype=str, delimiter=',')
	instances, labels = splitDataset(data)

	if args.mode=='erm':
		ERM(instances, labels, args.num_epochs, learning_rate)
	else:
		crossValidation(instances, labels, args.num_epochs, learning_rate)


if __name__ == '__main__':
	main()