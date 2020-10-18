


#randomforestregressor
def RandomForestRegressorCervical(train_x, train_y, x_test):
       from sklearn.ensemble import RandomForestClassifier
       regressor = RandomForestClassifier(n_estimators=100, random_state=0)
       regressor.fit(train_x, train_y)
       return regressor


'''
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
cancer_preds = forest_model.predict(val_X)
'''
