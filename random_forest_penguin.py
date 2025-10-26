import ydf
import numpy as np
import pandas as pd

data  = pd.read_csv('/home/apsit/Downloads/penguins.csv')

print(data.head())

np.random.seed(1)
is_test = np.random.rand(len(data)) < 0.2

train_data = data[~is_test]
test_data = data[is_test]

print("Training samples: ", len(train_data))
print("Testing samples: ", len(test_data))

model = ydf.RandomForestLearner(label='species').train(train_data)
model.plot_tree().to_file(path='/home/apsit/Downloads/penguins.html')


train_evaluation = model.evaluate(train_data)
test_evaluation = model.evaluate(test_data)

print("train accuracy: ",train_evaluation.accuracy)
print("test accuracy: ",test_evaluation.accuracy)
