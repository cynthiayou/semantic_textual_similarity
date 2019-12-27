# -*- coding: utf-8 -*-
import pickle
import numpy as np
if __name__ == '__main__':
    
    with open('models/model_rf.pickle', 'rb') as f:
        model_rf = pickle.load(f)
    
    with open('models/model_xg.pickle', 'rb') as f:
        model_xg = pickle.load(f)
    
    with open('processedData/test_data.pickle', 'rb') as f:
        X_test, test_ids = pickle.load(f)  
    
    y_pred_test_rf = model_rf.predict(X_test) 
    y_pred_test_xg = model_xg.predict(X_test) 

    
    # Generate the prediction file for dev set (Using just random forest model)
    y_pred_test_rf_round = np.round(y_pred_test_rf).astype('int')
    with open("test_prediction_rf.txt", "w") as f:
        f.write('id' + '\t' + 'Gold Tag' + '\n')
        for i in range(len(test_ids)):
            f.write(test_ids[i] + '\t' + str(y_pred_test_rf_round[i])+ "\n")  
            
            
    # Generate the prediction file for dev set (Using embedded model)        
    y_pred_test_embedded = np.round((y_pred_test_rf + y_pred_test_xg) / 2).astype('int')
    with open("test_prediction_embedded.txt", "w") as f:
        f.write('id' + '\t' + 'Gold Tag' + '\n')
        for i in range(len(test_ids)):
            f.write(test_ids[i] + '\t' + str(y_pred_test_embedded[i])+ "\n")  
             

    
    
    