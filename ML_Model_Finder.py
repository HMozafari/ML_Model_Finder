from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import RNN, LSTM, Activation, TimeDistributed, Dropout, LSTMCell, SimpleRNN
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from matplotlib import pyplot
from pandas.tools.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression, RFE
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import featuretools as ft
import autosklearn
import autosklearn.regression
import config
# include standard modules
import argparse

from autosklearn.metrics import accuracy, mean_squared_error
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
import multiprocessing
import shutil


def dataVisualization(data, target_feat):
### Univariate Histograms
#     pyplot.rcParams.update({'font.size': 2})
#     data.hist()
#     pyplot.rcParams.update({'font.size': 2})
#
#     pyplot.yticks(rotation=90)
#     pyplot.savefig('Univr-Hist.eps', format='eps')
#     pyplot.show()
#
#     pyplot.close()
# #########################
#
# ### Box and Whisker Plots ###
#     pyplot.rcParams.update({'font.size': 2})
#     data.plot(kind='box', subplots=True, layout=(int(len(list(data.columns.values))/2),int(len(list(data.columns.values))/2)), sharex=False, sharey=False)
#
#
#     pyplot.yticks(rotation=90)
#     pyplot.savefig('Box-Whis.eps', format='eps')
#     pyplot.show()
#
#
#     pyplot.close()
#########################

    ### Correlation Matrix Plot ###

    pyplot.rcParams.update({'font.size': 2})

    ix = abs(data.corr()).sort_values(target_feat, ascending=False).index
    data = data.loc[:, ix]
    names = data.columns.values
    correlations = abs(data.corr())
    print(data.corr())
    # plot correlation matrix
    fig = pyplot.figure()

    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(list(data.columns.values)), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)

    pyplot.suptitle('correlation')

    pyplot.xticks(rotation=90)
    pyplot.savefig('Feat_Cross_Corr-Matrix.eps', format='eps')
    pyplot.show()
    pyplot.close()
    #########################

    ### Scatter Plot ###
    pyplot.rcParams.update({'font.size': 2})
    scatter_matrix(data)

    pyplot.yticks(rotation=90)
    pyplot.savefig('Scatt-Plot.eps', format='eps')
    pyplot.show()
    pyplot.close()


#########################


def report_error(predictions, y_infer_actual_yield):
    # Use the forest's predict method on the test data
    # predictions = model.predict(X_infer)
    # Calculate the absolute errors
    errors = abs(predictions[:-1] - y_infer_actual_yield[:-1])  # mean absolute error on prediction.
    mae = np.mean(errors)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error on the model when it is compared against actual yield value:', round(np.mean(errors), 2),
          'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_infer_actual_yield[
                           :-1])  # since the last row does not have the actual yield (it corresponds to the current week)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
    return mae, accuracy


def plot(y_infer_actual_yield, inferDataSetIndex, modelAccuracy, growerAccuracy, predictions, y_infer_pred_yield):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    x = np.array(range(0, len(y_infer_actual_yield)))
    plt.xticks(x, list(inferDataSetIndex))
    actual_line, = plt.plot(x, y_infer_actual_yield, label='actual')
    pred_label = "algorithm (mape:" + str(round(modelAccuracy, 2)) + "%)"
    pred_line, = plt.plot(x, predictions, label=pred_label)
    manual_label = "manual (mape:" + str(round(growerAccuracy, 2)) + "%)"
    manual_line, = plt.plot(x, y_infer_pred_yield, label=manual_label)
    plt.legend(handles=[actual_line, pred_line, manual_line])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    fig.savefig('./yield_chart.png')
    plt.close(fig)


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)



def feature_selection(dataset, target_feature='actual_yield', feature_select_method='Most-Corr'):
    # this function returns the sorted features (their labels, names) based on the feature selection method that is defined as the input of this module.
    X = dataset.drop(labels=target_feature, axis=1)
    y = dataset[target_feature]

    X = X.dropna(axis=1)

    if (feature_select_method == 'Most-Corr'):
        try:
            ix = abs(dataset.corr()).sort_values(target_feature, ascending=False).index

        except KeyError:
            print("ERROR: Probably there is a nan or a VALUE error in your features columns.")
            raise
    elif (feature_select_method == 'UniVar-Select'):

        test = SelectKBest(score_func=f_regression, k=len(X.columns))

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(X.isnull().any(axis=0) )
        print(np.any(np.isnan(X.values)))
        print(np.all(np.isnan(X.values)))
        fit = test.fit(X, y)

        fit.transform(X)

        idxs_selected = fit.scores_

        idxs_selected = np.vstack([idxs_selected, [range(0, len(idxs_selected))]])
        idxs_selected = np.transpose(idxs_selected)
        idxs_selected = idxs_selected[idxs_selected[:, 0].argsort()]
        idxs_selected = np.transpose(idxs_selected)

        features_names = X.columns.values
        sorted_features_names = [target_feature]
        for idx in reversed(idxs_selected[1, :]):
            sorted_features_names.append(features_names[int(idx)])

        ix = sorted_features_names
    elif (feature_select_method == 'Rec-Feat-Elimin'):

        model = Ridge()
        rfe = RFE(model, 1)

        fit = rfe.fit(X, y)
        print(fit.support_)
        idxs_selected = fit.ranking_

        features_names = X.columns.values
        idxs_selected = np.vstack([features_names, idxs_selected])
        idxs_selected = np.transpose(idxs_selected)
        idxs_selected = idxs_selected[idxs_selected[:, 1].argsort()]
        idxs_selected = np.transpose(idxs_selected)

        sorted_features_names = list(reversed(idxs_selected[0, :]))
        ix = sorted_features_names
    elif (feature_select_method == 'Feat-Import'):

        trees = 250

        max_feat = len(X.columns.values)

        max_depth = 30

        min_sample = 2

        FeatImprSelcModel = RandomForestRegressor(n_estimators=trees,

                                                  max_features=max_feat,

                                                  max_depth=max_depth,

                                                  min_samples_split=min_sample,

                                                  random_state=0,

                                                  n_jobs=-1)

        FeatImprSelcModel.fit(X, y)
        feature_list = X.columns.values
        # Get numerical feature importances
        importances = list(FeatImprSelcModel.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in
                               zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
        imprFeatNames = [target_feature]
        for imprFeat in feature_importances:
            imprFeatNames.append(imprFeat[0])
        imprFeatNames

        ix = imprFeatNames

    return list(ix)


def feature_engineering(dataset, num_feat_to_select_and_comb=2, opr_list = ['Add', 'Sub', 'Mult', 'Div', 'Sin', 'Cos', 'Exp']):
    # this module receives a dataset and combines features to engineer more features.
    # if the num_feat_to_select_and_comb = None, then it automatically combines the features that are the most correlated (based on pierson correlation);
    # otherwise it selects num_feat_to_select_and_comb from all the features and make all the combinations based on these methods:
    # 1) multiplication
    # 2) Log-Multi

    new_dataset = pd.DataFrame([],index=dataset.index)
    if "Sin" in opr_list:
        for label in dataset.columns:
            new_label = "sin_" + str(label)
            new_dataset[new_label] = np.sin(dataset[label])
    if "Cos" in opr_list:
        for label in dataset.columns:
            new_label = "cos_" + str(label)
            new_dataset[new_label] = np.cos(dataset[label])
    if "Exp" in opr_list:
        for label in dataset.columns:
            new_label = "exp_" + str(label)
            new_dataset[new_label] = np.exp(dataset[label])

    dataset = pd.concat([dataset, new_dataset], axis=1)
    tuples = list(combinations(dataset.columns, num_feat_to_select_and_comb))
    for tuple in tuples:
        if "Add" in opr_list:
            new_add_label = str(tuple[0]) + str('_Add_') + str(tuple[1])
            new_add_data = np.add(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values))
            dataset[new_add_label] = new_add_data
        if "Sub" in opr_list:
            new_subt_label = str(tuple[0]) + str('_Min_') + str(tuple[1])
            new_subt_data = np.subtract(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values))
            dataset[new_subt_label] = new_subt_data

        if "Mult" in opr_list:
            new_Mult_label = str(tuple[0]) + str('_Mult_') + str(tuple[1])
            new_Mult_data = np.multiply(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values))
            dataset[new_Mult_label] = new_Mult_data

        if "Div" in opr_list:
            new_Div_label = str(tuple[0]) + str('_Div_') + str(tuple[1])
            new_Div_data = np.divide(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values))
            dataset[new_Div_label] = new_Div_data

    dataset_with_new_comb_feats = dataset


    return dataset_with_new_comb_feats


def odin(dataset, target_feature='actual_yield', export_corr_hist='True', num_of_most_corr_feats=10,
         feature_engineering=False,
         feature_select_methods=['Most-Corr', 'UniVar-Select', 'Rec-Feat-Elimin', 'Prin-Comp-Anal', 'Feat-Import']):
    # This module receives a Dataset and a target_feature
    # 1) it produces ABS-corr histogram against the target_feature.
    # 2) it return the num_of_most_corr features
    # 3) it generates a histogram file if export_corr_hist = True.
    # 4) if feature_engineering is true,
    #    4-1) it generates different combinations of columns
    dataset = dataset.dropna(axis=1)
    if (dataset[target_feature].isnull().values.any()):
        raise ValueError('There is a NaN in the target column. We cannot fill out those NaN using average.')
    dataset = dataset.fillna(dataset.mean())  # fill out any NaN in feature columns.
    # dataset = dataset.dropna()
    dataset = dataset.astype(float)

    print("data_visul_flag", export_corr_hist)

    if (export_corr_hist == True):
        dataVisualization(data=dataset, target_feat=target_feature)
        print("\n Data visualization has ended. \n")

    most_corr_sorted_labels = feature_selection(dataset=dataset, target_feature=target_feature,
                                                feature_select_method=feature_select_methods)
    print("most corr features are: \n", str(most_corr_sorted_labels))

    return most_corr_sorted_labels[0:num_of_most_corr_feats - 1]


def autoML_find_best_model(train_X_labels, train_y_label, train_dataset, test_X, n_cores=3,
                           processing_time_in_secs=3000):
    train_X = train_dataset[train_X_labels]
    train_y = train_dataset[train_y_label]
    feature_types = (['numerical'] * len(train_X_labels))

    # processes = []
    # spawn_reg = get_spawn_reg(train_X,train_y)
    # for i in range(int(n_cores/2)): # set this at roughly half of your cores
    #    p = multiprocessing.Process(target=spawn_reg, args=(i, 'digits'))
    #    p.start()
    #    processes.append(p)
    # for p in processes:
    #    p.join()

    automl = autosklearn.regression.AutoSklearnRegressor(
        ensemble_size=1,
        initial_configurations_via_metalearning=0,
        # include_preprocessors=["no_preprocessing"],  # in a case that we do not autoML does any feature-engineering
        time_left_for_this_task=processing_time_in_secs,
        per_run_time_limit=int(processing_time_in_secs / 10),
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 7},
        seed=1,
    )

    automl.fit(train_X.copy(), train_y.copy(), dataset_name='nature_fresh_lag1',
               feat_type=feature_types)
    automl.refit(train_X.copy(), train_y.copy())

    print(automl.show_models())

    return automl


def h2o_design_space_search(train_X_labels, train_y_label, train_dataset, test_X, max_runtime_secs):
    seed = 7
    np.random.seed(seed)
    h2o.init(max_mem_size_GB=3)

    # Import a sample binary outcome train/test set into H2O
    # train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
    # test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

    # Identify predictors and response
    x = train_X_labels
    y = train_y_label
    # x.remove(y)

    # For binary classification, response should be a factor
    # train[y] = train[y].asfactor()
    # test[y] = test[y].asfactor()
    train_dataset = h2o.H2OFrame(train_dataset)
    # Run AutoML for 30 seconds
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        nfolds=5
    )
    aml.train(x=x,
              y=y,
              training_frame=train_dataset)

    # View the AutoML Leaderboard
    lb = aml.leaderboard

    #print(lb)

    # print ("best model params are:" + str(lb.) )

    # my_training_frame = aml.actual_params['training_frame']
    # col_used = h2o.get_frame(my_training_frame)
    # print(col_used.columns)

    # The leader model is stored here
    #aml.leader

    # If you need to generate predictions on a test set, you can make
    # predictions directly on the `"H2OAutoML"` object, or on the leader
    # model object directly

    test_X = h2o.H2OFrame(test_X)
    #preds = aml.predict(test_X)

    # or:
    preds = aml.leader.predict(test_X)
    preds = [float(x) for x in
             preds.get_frame_data().split(sep='\n')[1:len(preds.get_frame_data().split(sep='\n')) - 1]]
    # preds = np.array(preds) - (0.9 * lb[1, 5]) # minuse the yield values from their MAE
    print("predicted values before manipulating them in rising edges:" + str(preds))
    new_preds = preds
    # new_preds = []
    # old_pred = preds[0]
    # new_preds.append(old_pred)
    # for pred in preds[1:]:
    #     new_pred = pred
    #     if (old_pred < new_pred): # there is a ascending in the yield value
    #         new_preds.append( 1.05 * (((new_pred- old_pred)/old_pred)+1) * new_preds[-1])
    #     else:
    #         new_preds.append(new_pred)
    #     old_pred = new_pred
    return [lb[1, 5], np.array(new_preds)]


def standardization(X):
    # Since the standardization function returns a list, we take the column as well as row names to add them to
    # the standardized list to make it to a DataFrame.
    X = pd.DataFrame(X)
    columnNames = list(X.columns.values)

    indexes = list(X.index.values)
    scaler = preprocessing.StandardScaler().fit(X)
    tempX = scaler.transform(X)

    return pd.DataFrame(tempX, index=indexes, columns=columnNames)


def main():


    if args.train_set_size:
        train_precental = args.train_set_size
    else:
        train_precental = config.train_set_precental

    if args.dataset_file_add:
        dataset_file_add = args.dataset_file_add
    else:
        dataset_file_add = config.dataset_file_add

    if args.oprs_for_feat_eng:
        oprs_for_feat_eng = args.oprs_for_feat_eng
    else:
        oprs_for_feat_eng = config.oprs_for_feat_eng

    if args.max_run_time:
        max_run_time = args.max_run_time
    else:
        max_run_time = config.max_run_time

    if args.num_of_most_corr_feats:
        num_of_most_corr_feats = args.num_of_most_corr_feats
    else:
        num_of_most_corr_feats = config.num_of_most_corr_feats

    if args.feature_select_methods:
        feature_select_methods = args.feature_select_methods
    else:
        feature_select_methods = config.feature_select_methods

    if args.verbose:
        verbose = args.verbose
    else:
        verbose = config.verbose

    if args.model_finder:
        model_finder = args.model_finder
    else:
        model_finder = config.model_finder

    if args.dataset_stat_visual_flag:
        dataset_stat_visual_flag = args.dataset_stat_visual_flag
    else:
        dataset_stat_visual_flag = config.dataset_stat_visual_flag

    # load dataset
    if (verbose > 0):
        print("###########################################")
        print ("datset is read from" + str(dataset_file_add) + ".")
        print("###########################################")
    dataset = pd.read_excel(dataset_file_add, sheetname='entire_data')
    if (verbose > 5):
        print (dataset)
    elif (verbose > 0 and verbose <=5):
        print ("dataset column names are:", str (dataset.columns.values))

    # based on the input file format there should be two first columns in the dataset_file that are translated into indeces (refer to the example input file).
    dataset = dataset.set_index(['indx', 'week_num'])

    # remove any empty column from dataset.
    dataset = dataset.dropna(axis=1)

    # check to see if there are some null values in the target parameter ('actual_yield') if there are any, raise an exception.
    if (dataset['actual_yield'].isnull().values.any()):
        raise ValueError('There is a NaN in the target column. We cannot fill out those NaN using average.')

    # fill out any NaN in feature columns with the mean of that column.
    dataset = dataset.fillna(dataset.mean())

    # make sure that the type of all entries in your dataset are float not string or boolean since they are used in ML training.
    dataset = dataset.astype(float)

    # get a copy of the 'actual_yield' and 'projected_yield' before normalization (standardization).
    # we do not want to Norm target parameters.
    actual_yield = dataset['actual_yield']
    projected_yield = dataset['projected_yield']

    # standardize all the dataset including target param (actual_yield).
    dataset = standardization(dataset)

    # do feature engineering on the input dataset
    # remove actual_yield and projected_yield from the features that we do feat_eng on them.
    # we shouldn't rely on 'projected_yield' as an input feat to our yield prediction models.
    feat_engineered_dataset = feature_engineering(
                                                  dataset=dataset.drop(['actual_yield', 'projected_yield'], axis=1),
                                                  num_feat_to_select_and_comb=2,
                                                  opr_list=oprs_for_feat_eng
                                                 )

    # for testing when we disable feature_engineeing
    #dataset_with_new_comb_feats = dataset

    # receive feature engineered dataset which is standardized and replace standardized "actual_yield" with its real values
    # note that we do not want to predict standardized target values ("actual_yield"). We want to measure MAE and RMSE which requires real values for "actual_yield"
    feat_engineered_dataset['actual_yield'] = actual_yield.values

    # select the "num_of_most_corr_feats" most correlated features based on "feature_select_methods" from the feature engineered dataset.
    # also odin save the correlation map onto a file.
    if (verbose>0):
        print("################################")
        print("Data Statis Visualization and feature selection have started....")
    print("data visul flag ", dataset_stat_visual_flag)
    name_of_most_corr_feat = odin(dataset=feat_engineered_dataset, target_feature='actual_yield',
                                  export_corr_hist=dataset_stat_visual_flag, num_of_most_corr_feats=num_of_most_corr_feats, feature_select_methods=feature_select_methods)
    if (verbose >0):
        print ("the labels for the most corr features are:", str (name_of_most_corr_feat[1:]))


    dataset = feat_engineered_dataset[name_of_most_corr_feat]
    del feat_engineered_dataset # to release memory

    #dataset_with_new_comb_feats = standardization(dataset_with_new_comb_feats)
    list_of_impr_feat_for_H2O = name_of_most_corr_feat[1:]  # we exclude actual_yield from the feature's list

    # dataset = dataset.drop(['projected_yield'],axis=1)
    #train_dataset = dataset[0:int(train_precental * len(dataset))]
    #test_dataset = dataset[int(train_precental * len(dataset)):]

    #listOfTupleNames = list(combinations(list_of_impr_feat_for_H2O, len(list_of_impr_feat_for_H2O)))
    features_and_their_MAE = 30

    # dataVisualization(data=dataset)

    # dataset = dataset.drop(['projected_yield'],axis=1)
    train_dataset = dataset[0:int(train_precental * len(dataset))]

    dataset['projected_yield'] = projected_yield
    test_dataset = dataset[int(train_precental * len(dataset)):]


    #train_X = train_dataset[list(tuple)]
    #train_y = train_dataset['actual_yield']

    test_X = test_dataset[list(list_of_impr_feat_for_H2O)]

    test_y = test_dataset['actual_yield']
    test_manual_preds = test_dataset['projected_yield']
    testSetIndexes = test_dataset.index


    if (model_finder == "H2O"):
        [best_model_MAE, pred] = h2o_design_space_search(train_X_labels = list_of_impr_feat_for_H2O,
                                                     train_y_label = 'actual_yield',
                                                     train_dataset=train_dataset,
                                                     test_X = test_X,
                                                     max_runtime_secs = max_run_time
                                                     )
    elif(model_finder == "AutoML"):
        autoML_best_model = autoML_find_best_model(train_X_labels=list_of_impr_feat_for_H2O,
                                               train_y_label='actual_yield',
                                               train_dataset=train_dataset,
                                               test_X=test_X,
                                               processing_time_in_secs=max_run_time
                                               )

        pred = autoML_best_model.predict(test_X)

    [mae, model_acc_on_test] = report_error(predictions=np.array([float(x) for x in pred]),
                                            y_infer_actual_yield=np.array([float(x) for x in test_y]))
    #pred = pred - np.array(0.9 * mae)

    [mae, model_acc_on_test] = report_error(predictions=np.array([float(x) for x in pred]),
                                            y_infer_actual_yield=np.array([float(x) for x in test_y]))

    [manual_mae, manual_acc] = report_error(predictions=np.array([float(x) for x in test_manual_preds]),
                                            y_infer_actual_yield=np.array([float(x) for x in test_y]))

    print("Actual Yield" + str(test_y))
    print("Pred Yield" + str(pred))

    plot(y_infer_actual_yield=np.array([float(x) for x in test_y]), inferDataSetIndex=testSetIndexes,
         modelAccuracy=model_acc_on_test,
         growerAccuracy=manual_acc, predictions=np.array([float(x) for x in pred]),
         y_infer_pred_yield=np.array([float(x) for x in test_manual_preds]))


    features_and_their_MAE.append([best_model_MAE, model_acc_on_test, list(tuple)])



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser(prog='PROG', add_help=True)

    parser.add_argument("--train_set_size", action="store", dest ="train_set_size", type=float,
                        help="This argument determines the percentage of dataset that is dedicated to train_set (e.g., 0.9 means the first 90 precent chunk of dataset is dedicated to training set).")
    parser.add_argument("--dataset_file_addr",  action="store", dest= "dataset_file_add", type=str,
                        help="This should point to a climate/yield file that you can find an example file in the current folder. note that the sheet-name has to be entire_data.")
    parser.add_argument("--oprs_for_feat_eng",  action="store", dest="oprs_for_feat_eng", type=str,
                        help="This argument determines the operations that should be used in feat engineering (e.g., \"[\"Add\", \"Sub\", \"Exp\", \"Sin\", \"Cos\", \"Mult\", \"Div\"]\", or \"mult\").\n Be careful about quotations when you make the list of feat_engin operations.")
    parser.add_argument("--max_run_time",  action="store", dest="max_run_time", type=int,
                        help="This argument determines the time that we want to spend on model finding/tuning (e.g., 3600, means one hour).")
    parser.add_argument("--num_of_most_corr_feats",  action="store", dest="num_of_most_corr_feats", type=int,
                        help="This argument determines the number of most important features (parameters) that are selected from feature engineered dataset (e.g., 20).")
    parser.add_argument("--feature_select_methods",  action="store", dest="feature_select_methods", type=str,
                        help="This argument determines the feature selection method (it could be ""feat-import"", ""rec-feat-elimin"", ""univar-select"", or ""most-corr"").")
    parser.add_argument("--verbose", '-v', action="store", dest="verbose", type=int,
                        help="This argument determines the verbosity level (it could be 0, 5, 10).")
    parser.add_argument("--model_finder",  action="store", dest="model_finder", type=str,
                        help="This argument determines which automated model finder algorithm should be deployed, automl, or h20.")
    parser.add_argument("--dataset_stat_visual_flag", type=str2bool, nargs='?',dest="dataset_stat_visual_flag",
                        help="This argument determines if statist graphs from dataset values (i.e., dist for each param, Pearson corr-map, etc.) should be generated and saved into files.")


    # read arguments from the command line
    args = parser.parse_args()

    main()