import argparse
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def splitDataset(data):

	columns = data[0]

	labels = []
	ins = []

	count = 0
	for row in data:
		if count!=0:
			ins.append(row[0:(len(row)-1)])
			if row[-1]=='0':
				labels.append('-1')
			else:
				labels.append(row[-1])
		else:
			count += 1

	# Convert each instance from string to float
	instances = []
	for i in ins:
		instances.append(list(map(float, i)))

	labels = list(map(int, labels))
	
	return instances, labels

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


def findDecisionStump(instances, labels, weights):
	erm_loss = sys.maxsize
	columnIndex = -1
	threshold = -1

	for col in range(len(instances[0])):
		
		column = instances[:, col]

		sorted_cols = []
		for val, label, weight in zip(column, labels, weights):
			sorted_cols.append([val, label, weight])
		
		sorted_cols.sort()
		loss = 0	

		for i in range(len(sorted_cols)):
			if sorted_cols[i][1]==1:
				loss += weights[i]

		if loss < erm_loss:
			erm_loss = loss
			threshold = sorted_cols[0][0] - 1
			columnIndex = col


		for i in range(len(sorted_cols)):
			loss -= (sorted_cols[i][1] * sorted_cols[i][2])

			if loss < erm_loss and i<len(sorted_cols)-1 and sorted_cols[i][0] != sorted_cols[i+1][0]:
				erm_loss = loss
				threshold = (sorted_cols[i][0] + sorted_cols[i+1][0])/2
				columnIndex = col

	return columnIndex, threshold

def check(numer, denom):
	if denom!=0:
		return (numer/denom)
	else:
		return 0

def difference(lt1, lt2):

	count = 0
	for i, j in zip(lt1, lt2):
		# print ('i:' , i)
		# print ('j: ', j)
		# print ()
		if i!=j:
			count += 1
	return count

def train(columns, instances, labels, rounds):
	
	numpyInstances = np.array(instances)
	weights = [1/len(numpyInstances)]* len(numpyInstances)

	learners = []

	erm_losses = []
	validation_losses = []

	for _ in range(rounds):
		error = 0

		columnIndex, threshold = findDecisionStump(numpyInstances, labels, weights)
		# print ('Column: ', columns[columnIndex], 'threshold: ', threshold)

		predicted_labels = []
		for i in range(len(instances)):
			if numpyInstances[i, columnIndex] <= threshold:
				predicted_label = -1
			else:
				predicted_label = 1
		
			predicted_labels.append(predicted_label)

			if labels[i]!=predicted_label:
				error += weights[i]

		learnerWeight = 0.5 * math.log( check(1 - error, error) + 0.00000001)

		learners.append({columnIndex : [threshold, learnerWeight]})

		weightsNorm = 0
		for i in range(len(instances)):
			weightsNorm += weights[i] * math.exp(-learnerWeight * labels[i] * predicted_labels[i])

		for i in range(len(instances)):
			weights[i] = (weights[i] * math.exp(-learnerWeight * labels[i] * predicted_labels[i]) ) / weightsNorm

		# if mode=='kfold':
		# 	erm_losses.append(test(learners, instances, labels))
		# 	validation_losses.append(test(learners, test_inputs, test_labels))

	return learners, weights
	

def test(learners, test_instances, test_labels):

	loss = 0
	for sample, label in zip(test_instances, test_labels):
		weightedSum = 0
		for learner in learners:
			for columnIndex in learner.keys():
				if sample[columnIndex] <= learner[columnIndex][0]:
					pd_l = -1
				else:
					pd_l = 1

				weightedSum += learner[columnIndex][1] * pd_l
		
		if np.sign(weightedSum)>=0:
			predicted_label = 1
		else:
			predicted_label = -1


		loss += lossFunc(label, predicted_label)

	erm_loss = loss/len(test_instances)
	return erm_loss
	

def Mode(columns, instances, labels, rounds, mode):
	
	if mode=='erm':
		learners, weights = train(columns, instances, labels, rounds)
		erm_loss = test(learners, instances, labels)
		print ('Weights: ', weights)
		print ('erm_loss: ', erm_loss)
		print ('Accuracy: ', 1 - erm_loss)

	else:
		validation_losses = []
		erm_losses = []
		folds = 10

		for f in range(rounds):
			inputs, outputs = kfold(instances, labels, folds)
			avg_fold_loss = 0
			erm_loss = 0
			for i in range(folds):
				test_inputs = inputs[i]
				test_labels = outputs[i]
				train_inputs = []
				train_labels = []
				
				for j in range(folds):
					if j!=i:
						train_inputs.extend(inputs[j])
						train_labels.extend(outputs[j])

				learners, weights = train(columns, train_inputs, train_labels, rounds)

				erm_loss += test(learners, train_inputs, train_labels)

				fold_loss = test(learners, test_inputs, test_labels)
				
				if mode!='plot' and f==rounds-1:
					print ('Fold ', i+1, ' Loss: ', fold_loss)
				
				avg_fold_loss += fold_loss

			erm_losses.append(erm_loss/folds)
			validation_losses.append(avg_fold_loss/folds)

		
		if mode!='plot':
			print ('Weights: ', weights)
			print ('Mean Fold Error: ', avg_fold_loss/folds)
			print ('Accuracy: ', 1 - (avg_fold_loss/folds))

		if mode=='plot':
			plot(validation_losses, erm_losses)
		

def plot(validation_losses, erm_losses):
	rounds = np.arange(len(validation_losses))
	plt.plot(rounds, validation_losses, marker = 'o', linestyle = ':', label = 'validation_losses')
	plt.plot(rounds, erm_losses, marker = 'o', linestyle = ':', label = 'erm_losses')
	plt.title('ERM and Validation Error vs Rounds(' + str(len(rounds)) + ')')
	plt.xlabel('Number of Rounds')
	plt.ylabel('Losses')
	plt.legend()
	plt.show()


def lossFunc(label, predicted_label):
	if label==predicted_label:
		return 0
	else:
		return 1


def main():

	parser = argparse.ArgumentParser(description='Adaptive Boosting')
	parser.add_argument('--mode', type=str, help='erm or kfold or plot')
	parser.add_argument('--num-rounds', type=int, help='number of rounds')

	args = parser.parse_args()

	data = np.loadtxt('Breast_cancer_data.csv', dtype=str, delimiter=',')
	columns = data[0]
	instances, labels = splitDataset(data)

	Mode(columns, instances, labels, args.num_rounds, args.mode)

if __name__ == '__main__':
	main()