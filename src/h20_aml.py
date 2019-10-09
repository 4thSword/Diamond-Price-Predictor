#Imports
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_squared_error

# Gloal constants:
TRAIN_PATH = "../input/processed_data.csv"
TEST_PATH = "../input/processed_test.csv"

# Functions definition:


# Execution:
if __name__ == "__main__":
    #Init H2O engine:
    h2o.init()

    # Import Data from previous processed *.csv files
    train = h2o.import_file(TRAIN_PATH)
    test = h2o.import_file(TEST_PATH)

    # Identify predictors and response
    x = train.columns
    y = 'price'
    x.remove(y)
    x.remove(train['id'])
    x_test = test.drop(['id'],axis=1)

    # Model Initialization and training:
    max_models = 40
    seed = 1

    aml = H2OAutoML(max_models= max_models, seed=seed)
    aml.train(x=x, y=y, training_frame=train)

    #Predict result:
    y_train_pred = aml.predict(x_test)

    # Metrics:
    try:
        y_pred = aml.predict(train[x])
        y_pred = p_pred.as_data_frame(use_pandas=True)
        y_train = train[y].as_data_frame(use_pandas=True)
        rmse = mean_squared_error(y_train,y_pred)
        print(rmse)
    except:
        rmse = "Not defined"
    #Output generation:
    submission = test['id']
    submission['Price'] = y_train_pred
    submission = submission.as_data_frame(use_pandas=True)

    submission.to_csv('../output/submission.csv',index=False)

    # Adding metrics to a log, for next study of better model.
    with open('../output/log.txt',"a+") as f: 
        f.write("RMSE: {} | MODEL: AutoML | MAX MODELS= {} | SEED= {} | COLUMNS: {} \n".format(rmse, max_models, seed, len(training.columns)))