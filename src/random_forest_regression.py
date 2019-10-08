#Imports
import pandas as pd
#from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gloal constants:
TRAIN_PATH = "../input/processed_data.csv"
TEST_PATH = "../input/processed_test.csv"

# Functions definition:


# Execution:
if __name__ == "__main__":
    #Data preparing
    training = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)


    data_train = training.drop(['price'],axis = 1)
    data_submit = test.drop(['id'],axis = 1)    
    y = training['price']

    X_train, X_test, y_train, y_test = train_test_split(data_train,y, test_size=0.1)

    #Model Training:
    
    n_estimators = 100
    rfr = RandomForestRegressor(n_estimators=n_estimators)
    rfr.fit(X_train, y_train)

    #Applying trained model to our train set:
    y_test_pred = rfr.predict(X_test)
    #checking the error
    rmse=mean_squared_error(y_test,y_test_pred)
    print(rmse)

    # Applying model to our submission set:
    y_pred = rfr.predict(data_submit)
    


    #Result tratement to be submmited:
    submission = pd.DataFrame({
        'Id':test['id'],
        'Price': y_pred
    })
    #submission.Price = submission.Price.apply(lambda x: ((x**2)**(1/2)/10))

    
    # Generating output file:
    submission.to_csv('../output/submission.csv',index=False)

    # Adding metrics to a log, for next study of better model.
    with open('../output/log.txt',"a+") as f: 
        f.write("RMSE: {} | MODEL: MLPRegressor | N-ESTIMATORS= {} | COLUMNS: {} \n".format(rmse, n_estimators, len(training.columns)))