#import
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
#defining path and using pandas to read file
cancer_file_path = 'kag_risk_factors_cervical_cancer.csv'
cancer_data = pd.read_csv(cancer_file_path)
#check data by printing summary of csv file
print(cancer_data.describe())
#removing all rows with empty cells
cancer_data = cancer_data.dropna(axis=0)
#defining dependent variable
y = cancer_data.Biopsy
# defining independent variables
cancer_features = ['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
       'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
       'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
       'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
       'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
       'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
       'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
       'Citology']
X = cancer_data[cancer_features]
# decision tree
from sklearn.tree import DecisionTreeRegressor
cancer_model = DecisionTreeRegressor(random_state=1)
cancer_model.fit(X, y)
print("Making predictions for the following:")
print(X.head(n=20))
print("The predictions are")
print(cancer_model.predict(X.head(n=20)))
from sklearn.metrics import mean_absolute_error
predicted_cancer = cancer_model.predict(X)
print(mean_absolute_error(y, predicted_cancer))
# splitting data for testing
from sklearn.model_selection import train_test_split
#calculating MSE
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
cancer_model = DecisionTreeRegressor()
cancer_model.fit(train_X, train_y)
val_predictions = cancer_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
print(tree.plot_tree(cancer_model))
#creating graphical version of decision tree
'''
import graphviz
cervical_data = tree.export_graphviz(cancer_model, out_file=None)
graph = graphviz.Source(cervical_data)
graph.render("cervical")
cervical_data = tree.export_graphviz(cancer_model, out_file=None,
                     feature_names= cancer_features,
                     class_names= cancer_data.Biopsy,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(cervical_data)
graph
'''
#randomforestregressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
cancer_preds = forest_model.predict(val_X)
#print(cancer_preds)
#print(mean_absolute_error(val_y, cancer_preds))
#print(classification_report(val_y, val_predictions)

#printing metric values to assess models
cm = confusion_matrix(val_y, val_predictions)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
#Neural Net
class NeuralNet():
    class NeuralNet():

        def __init__(self, layers=[13, 8, 1], learning_rate=0.001, iterations=100):
            self.params = {}
            self.learning_rate = learning_rate
            self.iterations = iterations
            self.loss = []
            self.sample_size = None
            self.layers = layers
            self.X = None
            self.y = None

        def init_weights(self):
            np.random.seed(1)  # Seed the random number generator
            self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
            self.params['b1'] = np.random.randn(self.layers[1], )
            self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
            self.params['b2'] = np.random.randn(self.layers[2], )

        def relu(self, Z):
            '''
            The ReLu activation function is to performs a threshold
            operation to each input element where values less
            than zero are set to zero.
            '''
            return np.maximum(0, Z)

        def sigmoid(self, Z):
            '''
            The sigmoid function takes in real numbers in any range and
            squashes it to a real-valued output between 0 and 1.
            '''
            return 1.0 / (1.0 + np.exp(-Z))

        def forward_propagation(self):
            '''
            Performs the forward propagation
            '''

            Z1 = self.X.dot(self.params['W1']) + self.params['b1']
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            yhat = self.sigmoid(Z2)
            loss = self.entropy_loss(self.y, yhat)

            # save calculated parameters
            self.params['Z1'] = Z1
            self.params['Z2'] = Z2
            self.params['A1'] = A1

            return yhat, loss

        def back_propagation(self, yhat):
            '''
            Computes the derivatives and update weights and bias according.
            '''

            def dRelu(x):
                x[x <= 0] = 0
                x[x > 0] = 1
                return x

            dl_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
            dl_wrt_sig = yhat * (1 - yhat)
            dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

            dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
            dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
            dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

            dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
            dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
            dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

            # update the weights and bias
            self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
            self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
            self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
            self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

        def fit(self, X, y):
            '''
            Trains the neural network using the specified data and labels
            '''
            self.X = X
            self.y = y
            self.init_weights()  # initialize weights and bias

            for i in range(self.iterations):
                yhat, loss = self.forward_propagation()
                self.back_propagation(yhat)
                self.loss.append(loss)

        def predict(self, X):
            '''
            Predicts on a test data
            '''
            Z1 = X.dot(self.params['W1']) + self.params['b1']
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            pred = self.sigmoid(Z2)
            return np.round(pred)

        def acc(self, y, yhat):
            '''
            Calculates the accutacy between the predicted valuea and the truth labels
            '''
            acc = int(sum(y == yhat) / len(y) * 100)
            return acc

        def plot_loss(self):
            '''
            Plots the loss curve
            '''
            plt.plot(self.loss)
            plt.xlabel("Iteration")
            plt.ylabel("logloss")
            plt.title("Loss curve for training")
            plt.show()


















