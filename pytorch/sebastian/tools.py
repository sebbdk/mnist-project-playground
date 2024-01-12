import csv

def saveResult(epoch,learning_rate,loss,batch_size,epoch_pro,epoch_performance, name):
	with open(f'{name}.csv', 'w', newline='') as csvfile:
		resultwriter = csv.writer(csvfile)
						
		resultwriter.writerow([epoch]+[learning_rate]+[loss]+[batch_size])
		for pro,perf in zip(epoch_pro, epoch_performance):
			resultwriter.writerow([pro, perf])
	



if __name__ == "__main__":
	name = "test_data"
	epoch = 30
	learning_rate = 0.1
	loss = "MSE"
	batch_size = 30
	epoch_pro = [0,1,2,3,4]
	epoch_performance = [0,3,4,5,6]
	saveResult(epoch,learning_rate,loss,batch_size,epoch_pro,epoch_performance,name)