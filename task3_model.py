# -*- coding: utf-8 -*-
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  mean_squared_error
from scipy.stats import pearsonr
import pickle
import numpy as np
import gc


def model_random_forest(X_train, Y_train, X_dev, X_test):
    rf = RandomForestRegressor(n_jobs=-1, random_state = 10, oob_score = True)
    param_grid = {'n_estimators': [300]}
    model = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=None, iid=False, cv=10)
    model.fit(X_train, Y_train)
    print("best params: ")
    print(model.best_params_)
    print("best cv score: ")
    print(model.best_score_)
    y_pred_train = model.predict(X_train)
    y_pred_dev = model.predict(X_dev)
    y_pred_test = model.predict(X_test)
    
    #with open('models/model_rf.pickle', 'wb') as f:
        #pickle.dump(model, f)
    return y_pred_train, y_pred_dev, y_pred_test

def model_XG(X_train, Y_train, X_dev, X_test):
    clf = XGBRegressor(nthread = 4)
    clf.fit(X_train, Y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_dev = clf.predict(X_dev)
    y_pred_test = clf.predict(X_test)
   # with open('models/model_xg.pickle', 'wb') as f:
       # pickle.dump(clf, f)
    return y_pred_train, y_pred_dev, y_pred_test
    

if __name__ == '__main__':
    with open('processedData/train_data.pickle', 'rb') as f :
        X_train, Y_train = pickle.load(f)
    with open('processedData/dev_data.pickle', 'rb') as f :
        X_dev, Y_dev = pickle.load(f)
    with open('processedData/test_data.pickle', 'rb') as f:
        X_test, test_ids = pickle.load(f)
    
    
    y_pred_train_rf, y_pred_dev_rf, y_pred_test_rf = model_random_forest(X_train, Y_train, X_dev, X_test)
    print("the pearsonr of random forest for training data:", pearsonr(y_pred_train_rf, Y_train)[0])
    print("the pearsonr of random forest for dev data:", pearsonr(y_pred_dev_rf, Y_dev)[0])
    print("The MSE of random forest for training data: ", mean_squared_error(y_pred_train_rf, Y_train))
    print("The MSE of random forest for dev data: ", mean_squared_error(y_pred_dev_rf, Y_dev))    
    
    y_pred_train_xg, y_pred_dev_xg, y_pred_test_xg = model_XG(X_train, Y_train, X_dev, X_test)
    print("the pearsonr of xgboost for training data:", pearsonr(y_pred_train_xg, Y_train)[0])
    print("the pearsonr of xgboost for dev data: ", pearsonr(y_pred_dev_xg, Y_dev)[0])
    print("The MSE of xgboost for training data: ", mean_squared_error(y_pred_train_xg, Y_train))
    print("The MSE for dev data:", mean_squared_error(y_pred_dev_xg, Y_dev))
    
    print("the pearsonr of embedded models for training data:", pearsonr((y_pred_train_rf + y_pred_train_xg)/2, Y_train)[0])
    print("the pearsonr of embedded models for dev data:", pearsonr((y_pred_dev_rf + y_pred_dev_xg)/2, Y_dev)[0])
    print("The MSE of embedded models for training data: ", mean_squared_error((y_pred_train_rf + y_pred_train_xg)/2, Y_train))
    print("The MSE of embedded models for dev data: ", mean_squared_error((y_pred_dev_rf + y_pred_dev_xg)/2, Y_dev))
    
    with open('models/model_rf.pickle', 'rb') as f:
        model_rf = pickle.load(f)
    
    with open('models/model_xg.pickle', 'rb') as f:
        model_xg = pickle.load(f)
        
    y_dev_rf = model_rf.predict(X_dev) 
    y_dev_xg = model_xg.predict(X_dev) 
         
    # Generate the prediction file for dev set
    y_pred_dev_rf = np.round(y_dev_rf).astype('int')
    #y_pred_dev_embedded = np.round((y_dev_rf + y_dev_xg) / 2).astype('int')
    with open('processedData/dev_ids.pickle', 'rb') as f:
        dev_ids = pickle.load(f)
    with open("dev_prediction.txt", "w") as f:
        f.write('id' + '\t' + 'Gold Tag' + '\n')
        for i in range(len(dev_ids)):
            f.write(dev_ids[i] + '\t' + str(y_pred_dev_rf[i])+ "\n")  
             
        

    
    
    
        
    