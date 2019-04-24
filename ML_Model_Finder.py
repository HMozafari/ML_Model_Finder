# best command line parameters that are found so far:

# Agrimind:python3 ../ML_Model_Finder/ML_Model_Finder.py --dataset_file_addr ./thanos_output_agrimind_with_features_NIC_removed_without_yield_rows_completely.csv --oprs_for_feat_eng "["Add", "Sub"]" --max_run_time 1000 --num_of_most_corr_feats 160 --feature_select_methods Most-Corr --model_finder H2O --dataset_stat_visual_flag False --train_set_size 0.75 --rmv_feat_with_corr_higher_than 1 --slope 0.4 --bias 0 --target_feat "y7" --drop_feat "next_next_week_yield" --num_week_shift 0 --alr_gen_feats True --add_mov_win_rolling_feats False --match_with_actual_yield False
# SERA : python3 ../ML_Model_Finder/ML_Model_Finder.py --dataset_file_addr ./SERA_without_Features_Hasan_removed_rows_without_actual_yield.csv --oprs_for_feat_eng "["Add", "Sub"]" --max_run_time 400 --num_of_most_corr_feats 21 --feature_select_methods Most-Corr --model_finder H2O --dataset_stat_visual_flag False --train_set_size 0.75 --rmv_feat_with_corr_higher_than 1 --slope 0.4 --bias 0 --target_feat "y5" --drop_feat "next_next_week_yield" --num_week_shift 0 --alr_gen_feats True --add_mov_win_rolling_feats  False --match_with_actual_yield  False
# Rimato: python3 ../ML_Model_Finder/ML_Model_Finder.py --dataset_file_addr ./rimato1_weekly_total_df_with_feat20190417113907\ \(1\).csv --oprs_for_feat_eng "["Add"]" --max_run_time 800 --num_of_most_corr_feats 120 --feature_select_methods Most-Corr --model_finder H2O --dataset_stat_visual_flag False --train_set_size 0.75 --rmv_feat_with_corr_higher_than 0.93 --slope 0.4 --bias 0 --target_feat "y6" --drop_feat "next_next_week_yield" --num_week_shift 0 --alr_gen_feats True --add_mov_win_rolling_feats False --match_with_actual_yield False
# SanLucar : python3 ../ML_Model_Finder/ML_Model_Finder.py --dataset_file_addr ./sanlucar_weekly_total_df20190403231530.csv --oprs_for_feat_eng "["Add", "Sub"]" --max_run_time 400 --num_of_most_corr_feats 37 --feature_select_methods Most-Corr --model_finder H2O --dataset_stat_visual_flag False --train_set_size 0.75 --rmv_feat_with_corr_higher_than 1 --slope 0.4 --bias 0 --target_feat "y6" --drop_feat "next_next_week_yield" --num_week_shift 0 --alr_gen_feats True --add_mov_win_rolling_feats  False --match_with_actual_yield  False


import psutil
from scipy.ndimage.interpolation import shift
from scipy import stats
from scipy.stats import pearsonr
import getch
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import RNN, LSTM, Activation, TimeDistributed, Dropout, LSTMCell, SimpleRNN
import pandas as pd
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
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

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from autosklearn.metrics import accuracy, mean_squared_error
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
import multiprocessing
import shutil
from fancyimpute import KNN, IterativeImputer, NuclearNormMinimization, SoftImpute
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoLars
from sklearn.model_selection import ShuffleSplit, TimeSeriesSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import math

from multiprocessing import Process,Manager


def change_the_slope_of_graph(pred, slope):
    new_pred = []
    new_pred.append(pred[0])
    for i in range(1, len(pred)):
        curr_slope = pred[i] - pred[i - 1]
        new_pred.append(pred[i] + slope * curr_slope)
    return new_pred


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

    pyplot.rcParams.update({'font.size': 0.2})

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


def plot(fig_name, y_infer_actual_yield, inferDataSetIndex, modelAccuracy, growerAccuracy, predictions,
         y_infer_pred_yield):
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
    fig.savefig(fig_name)
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

def calc_corr_between_a_feat_and_target_feat(target_feat, dataset, my_queue):
    my_queue.put(dataset.corr().abs().loc[target_feat,:])

def feature_selection(dataset, num_cores, target_feature='actual_yield', feature_select_method='Most-Corr',
                      opr_list="""[Mult]""", num_procs=100
                      ):
    import ast
    #num_cores = 100
    opr_list = opr_list.replace('[', '')
    opr_list = opr_list.replace(']', '')

    opr_list = [x for x in map(str.strip, opr_list.split(',')) if x]

    # opr_list = ast.literal_eval(opr_list)
    print("The operations are:")
    print(opr_list)

    # this function returns the sorted features (their labels, names) based on the feature selection method that is defined as the input of this module.
    X = dataset.drop(labels=target_feature, axis=1)
    y = dataset[target_feature]

    X = X.dropna(axis=1)

    m = multiprocessing.Manager()
    my_queue = m.Queue()

    T = []



    pearson_corr_dict = {}
    if (feature_select_method == 'Most-Corr'):
        corr_dataframe = pd.DataFrame([])
        try:
            clmn_names = list(dataset.columns)
            clmn_names.remove(target_feature)
            for i in range(1, int((num_cores)/num_procs)+1):
                for j in range (1, num_procs+1):
                    cnt = (i-1)*num_procs + j
                    partial_clmn_names = clmn_names[int((cnt - 1) * (len(clmn_names) / num_cores)): int(
                        (cnt) * (len(clmn_names) / num_cores)) - 1]
                    partial_dataset = pd.concat([dataset[target_feature], dataset[partial_clmn_names]], axis=1)
                    p = Process(target=calc_corr_between_a_feat_and_target_feat,
                                args=(target_feature, partial_dataset, my_queue))
                    T.append(p)
                for t in T[ (i-1)*num_procs : i*num_procs-1]:
                    t.start()
                for t in T[ (i-1)*num_procs : i*num_procs-1]:
                    t.join()
                for t in T[ (i-1)*num_procs : i*num_procs-1]:
                    x = my_queue.get()
                    corr_dataframe = pd.concat([corr_dataframe, x], axis=0)

            # for clmn_label in dataset.columns:
            #     calc_corr_dataset = pd.DataFrame([])
            #     calc_corr_dataset[clmn_label] = dataset[clmn_label]
            #     calc_corr_dataset[target_feature] = dataset[target_feature]
            #     try:
            #         corr_dataframe[clmn_label] = pd.Series(
            #             abs(calc_corr_dataset[calc_corr_dataset.columns].corr()[target_feature][:-1]).iloc[0],
            #             corr_dataframe.index)
            #     except IndexError:
            #         continue
            corr_dataframe = corr_dataframe.drop_duplicates()
            corr_dataframe = pd.DataFrame(corr_dataframe.values, columns=['corr'], index=corr_dataframe.index)

            ix = corr_dataframe.sort_values(ascending=False, axis=0, by='corr').index[1:]
            # ix = abs(dataset[dataset.columns].corr()[target_feature][:-1]).sort_values(ascending=False).index
            # ix = abs(dataset.corr()).sort_values(target_feature, ascending=False).index

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

        # imprFeatNames

        ix = imprFeatNames

    ix = list(ix)

    # list_of_selected_feats = ix

    copyix = ix.copy()
    already_used_feats = []

    list_of_selected_feats = []

    for feat_label in copyix:
        more_than_two_feats_flag = False
        for split in opr_list:
            feats = feat_label.split(sep="_" + split + "_")
            if (len(feats) >= 2):  # there are more than two features.
                more_than_two_feats_flag = True

                break

        if more_than_two_feats_flag:
            flag = False
            for feat in feats:

                for already_used_feat in already_used_feats:
                    if feat in already_used_feat:
                        flag = True

            if (flag == False):
                list_of_selected_feats.append(feat_label)
                for feat in feats:
                    already_used_feats.append(feat)

        else:
            feats = feat_label
            flag2 = False
            for already_used_feat in already_used_feats:
                if feats in already_used_feat:
                    flag2 = True

            if (flag2 == False):
                list_of_selected_feats.append(feat_label)
                already_used_feats.append(feat_label)

        if (len(list_of_selected_feats) >= 1000):
            return list(list_of_selected_feats)

    return list(list_of_selected_feats)


def fillNaN_using_Reg(dataset):
    # This module takes a dataset and fill-out all the NaN using a regression model based on some columns that are fully full.
    partial_dataset_without_NaN = dataset.dropna(axis=1)  # keep the columns that do not have any NaN
    clmn_labels_containing_null_values = dataset.columns[dataset.isnull().any()]
    for clmn in clmn_labels_containing_null_values:
        clmn_with_null_value = pd.DataFrame(dataset[clmn], index=dataset.index)
        test_index = clmn_with_null_value[clmn_with_null_value.isnull()].index
        train_index = clmn_with_null_value[clmn_with_null_value.notnull()].index

        train_X = partial_dataset_without_NaN.loc[train_index, :]
        train_Y = clmn_with_null_value.loc[train_index, :]

        test_X = partial_dataset_without_NaN.loc[test_index, :]


def inteligent_shifted(shiftted_feat, target_feat, num_of_shift):
    intel_shifted_feat = []
    intel_shifted_feat_temp = []
    intel_shifted_feat = list(shiftted_feat.values[0:3])
    #intel_shifted_feat_temp = intel_shifted_feat
    for indx in range(3, len(shiftted_feat)-num_of_shift-1):

        backward = intel_shifted_feat + [shiftted_feat.values[indx-1]]
        inplace = intel_shifted_feat + [shiftted_feat.values[indx]]
        forward = intel_shifted_feat + [shiftted_feat.values[indx+1] ]


        try:
            backward_corr = abs(pearsonr(target_feat.values[0: indx+1], backward)[0])
            inplace_corr = abs(pearsonr(target_feat.values[0: indx+1], inplace)[0])
            forward_corr = abs(pearsonr(target_feat.values[0: indx+1], forward)[0])
            #forward_corr = 0
        except ValueError:
            print (ValueError)
            getch.getch()

        if (math.isnan(backward_corr)):
            backward_corr = 0
        if (math.isnan(inplace_corr)):
            inplace_corr = 0
        if (math.isnan(forward_corr)):
            forward_corr = 0

        if (backward_corr >= inplace_corr and backward_corr >= forward_corr ):
            intel_shifted_feat = intel_shifted_feat + [shiftted_feat.values[indx - 1]]
        elif (inplace_corr>= backward_corr and inplace_corr>= forward_corr ):
            intel_shifted_feat = intel_shifted_feat + [shiftted_feat.values[indx]]
        elif (forward_corr>= backward_corr and forward_corr>= inplace_corr ):
            intel_shifted_feat = intel_shifted_feat + [shiftted_feat.values[indx + 1]]
        else:
            intel_shifted_feat = intel_shifted_feat + shiftted_feat.values[indx]
    for i in range(1, num_of_shift+1):
        intel_shifted_feat.append(shiftted_feat.values[-1*num_of_shift-2 + i])

    return intel_shifted_feat + [shiftted_feat.values[-1]] # for the last two elements of the feat we cannot calculate the corr since there is no n+1 element.


def move_feats_and_add_new_feats(dataset, min_num_mov, max_num_mov, target_feat, intel_feat_moving):
    new_dataset = pd.DataFrame([])
    # try:
    #     dataset = dataset.drop(['actual'], axis=1)
    # except KeyError:
    #     dataset = dataset
    #
    # mov_dataset = pd.DataFrame([])
    # for label_name in dataset.columns:
    #     new_label = label_name + "_intel_move"
    #     mov_dataset[new_label] = dataset[label_name]
    #     mov_dataset.loc[:, new_label] =  inteligent_shifted(dataset[label_name],  dataset[target_feat], int(target_feat[1]))
    #
    # dataset = pd.concat([dataset, mov_dataset], axis=1)

    if (min_num_mov < 0):
        try:
            dataset = dataset.drop(['actual'], axis=1)
        except KeyError:
            dataset = dataset
    list_of_feat_except_target=list(dataset.columns)
    list_of_feat_except_target.remove(target_feat)
    for label in list_of_feat_except_target:

        if (label == "total_fruits_total"):
            print(label)
        two_clmn_dataset = pd.DataFrame([])
        two_clmn_dataset[target_feat] = dataset[target_feat].fillna(0)
        corr = -0.1
        # best_num_mov = 0
        best_num_mov_label = "Mov_Down_" + str('0') + "_" + str(label)
        for num_mov in range(min_num_mov, max_num_mov + 1):

            new_label = "Mov_Down_" + str(num_mov) + "_" + str(label)


            two_clmn_dataset[new_label] = dataset[label].shift(num_mov)
            two_clmn_dataset[new_label].fillna(0)
            #two_clmn_dataset = pd.DataFrame(SoftImpute(verbose=0).fit_transform(two_clmn_dataset),
            #                                index=two_clmn_dataset.index,
            #                                columns=two_clmn_dataset.columns.values)


            new_corr = abs(pearsonr(two_clmn_dataset[new_label].values[:num_mov - 1*int(target_feat[1])], two_clmn_dataset[target_feat].values[:num_mov - 1*int(target_feat[1])])[0])
            if (new_corr> corr):
                corr = new_corr
                best_num_mov_label = new_label


        num_mov = 0
        new_label = "Mov_Down_" + str(num_mov) + "_" + str(label)

        two_clmn_dataset[new_label] = dataset[label]
        two_clmn_dataset[new_label].fillna(0)
        #two_clmn_dataset = pd.DataFrame(SoftImpute(verbose=0).fit_transform(two_clmn_dataset),
        #                                index=two_clmn_dataset.index,
        #                                columns=two_clmn_dataset.columns.values)

        new_corr = abs(pearsonr(two_clmn_dataset[new_label].values[:num_mov - 1 * int(target_feat[1])],
                                two_clmn_dataset[target_feat].values[:num_mov - 1 * int(target_feat[1])])[0])
        if (new_corr > corr):
            corr = new_corr
            best_num_mov_label = new_label

        if (best_num_mov_label == ("Mov_Down_" + str('0') + "_" + str(label)) ):

            if(intel_feat_moving == True):
                #print ("dummy")
                try:
                    dataset = dataset.drop(label, axis=1)
                    two_clmn_dataset.loc[:, new_label] = inteligent_shifted(two_clmn_dataset[new_label],
                                                                            two_clmn_dataset[target_feat], 4*int(target_feat[1]))

                    new_dataset[best_num_mov_label + "_inteligent_feat_moving"] = two_clmn_dataset[new_label]
                    #print ("dummy")
                except ValueError:
                    two_clmn_dataset.loc[:, new_label] = two_clmn_dataset.loc[:, new_label]


        else:
            new_dataset[best_num_mov_label] = two_clmn_dataset[best_num_mov_label]

        #
        # two_clmn_dataset = pd.dataframe([])
        # two_clmn_dataset['next_week_yield'] = dataset['next_week_yield']
        #
        #
        # for num_mov in range(1, num_movement + 1, 2):
        #
        #     mov_down_label = "mov_down_" + str(num_mov) + "_" + str(label)
        #
        #     mov_downdown_label = "mov_down_" + str(num_mov) + "_down_" + str(num_mov) + "_" + str(label)
        #
        #     new_label = "comb_of_mov_down_" + str(num_mov) + "_" + "mov_down_" + str(num_mov+1) + "_" + str(label)
        #
        #     two_clmn_dataset[new_label] = dataset[label].shift(num_mov)
        #
        #     two_clmn_dataset[mov_down_label] = dataset[label].shift(num_mov)
        #
        #     two_clmn_dataset[mov_downdown_label] = dataset[label].shift(num_mov + 1)
        #
        #     two_clmn_dataset = pd.dataframe(softimpute(verbose=0).fit_transform(two_clmn_dataset),
        #                                     index=two_clmn_dataset.index,
        #                                     columns=two_clmn_dataset.columns.values)
        #
        #     for indx in range(num_mov, len(dataset.index.values) - num_mov + 1 ):
        #             two_clmn_dataset[mov_down_label][0:indx-1] = two_clmn_dataset[new_label][0:indx-1]
        #             two_clmn_dataset[mov_downdown_label][0:indx - 1] = two_clmn_dataset[new_label][0:indx - 1]
        #
        #
        #             if (abs(two_clmn_dataset['next_week_yield'][0:indx].corr(two_clmn_dataset[mov_down_label][0:indx])) >  abs(two_clmn_dataset['next_week_yield'][0:indx].corr(two_clmn_dataset[mov_downdown_label][0:indx])) ):
        #                 two_clmn_dataset[new_label][indx] = two_clmn_dataset[mov_down_label][indx]
        #             else:
        #                 two_clmn_dataset[new_label][indx] = two_clmn_dataset[mov_downdown_label][indx]
        #
        #     new_dataset[new_label] = two_clmn_dataset[new_label]

    new_dataset = pd.concat([dataset, new_dataset], axis=1)

    # new_dataset = new_dataset.T.drop_duplicates().T
    print("##################################################################################")
    print("##############################   at the end of moving features ###################")
    print("##################################################################################")
    # new_dataset = new_dataset.interpolate(method='linear', axis=1)
    # new_dataset = new_dataset.fillna(new_dataset.mean())
    return new_dataset


def generate_feat_engineered_feats(dataset, tuples, num_feat_to_select_and_comb, my_queue, opr_list, num_cores=8 ):
    if (num_feat_to_select_and_comb == 2):
        for tuple in tuples:
            if "Add" in opr_list:
                new_add_label = str(tuple[0]) + str('_Add_') + str(tuple[1])
                new_add_data = np.add(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values))
                dataset[new_add_label] = new_add_data

            if "Sub" in opr_list:
                new_subt_label = str(tuple[0]) + str('_Sub_') + str(tuple[1])
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

    if (num_feat_to_select_and_comb == 3):
        for tuple in tuples:
            if "Add" in opr_list:
                new_add_label = str(tuple[0]) + str('_Add_') + str(tuple[1]) + str('_Add_') + str(tuple[2])
                new_add_data = np.add(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                      np.array(dataset[tuple[2]].values))
                dataset[new_add_label] = new_add_data
            if "Sub" in opr_list:
                new_subt_label = str(tuple[0]) + str('_Sub_') + str(tuple[1])
                new_subt_data = np.subtract(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                            np.array(dataset[tuple[2]].values))
                dataset[new_subt_label] = new_subt_data

            if "Mult" in opr_list:
                new_Mult_label = str(tuple[0]) + str('_Mult_') + str(tuple[1])
                new_Mult_data = np.multiply(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                            np.array(dataset[tuple[2]].values))
                dataset[new_Mult_label] = new_Mult_data

            if "Div" in opr_list:
                new_Div_label = str(tuple[0]) + str('_Div_') + str(tuple[1])
                new_Div_data = np.divide(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                         np.array(dataset[tuple[2]].values))
                dataset[new_Div_label] = new_Div_data

    dataset_with_new_comb_feats = dataset
    my_queue.put(dataset_with_new_comb_feats)









def feature_engineering_parallely (dataset, num_feat_to_select_and_comb=2,
                        opr_list=['Add', 'Sub', 'Mult', 'Div', 'Sin', 'Cos', 'Exp'], num_cores = 8, num_procs =100):
    # this module receives a dataset and combines features to engineer more features.
    # if the num_feat_to_select_and_comb = None, then it automatically combines the features that are the most correlated (based on pierson correlation);
    # otherwise it selects num_feat_to_select_and_comb from all the features and make all the combinations based on these methods:
    # 1) multiplication
    # 2) Log-Multi
    new_dataset = pd.DataFrame([])
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
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    dataset = dataset.drop_duplicates()
    tuples = list(combinations(dataset.columns, num_feat_to_select_and_comb))

    m = multiprocessing.Manager()
    my_queue = m.Queue()


    temp_dataset = pd.DataFrame([])
    T = []
    #num_procs = 100
    for i in range(1, int((num_cores) / num_procs)+1):
        for j in range(1, num_procs+1):
            cnt = (i - 1) * num_procs + j

            partial_tuples = tuples[int((cnt - 1) * (len(tuples) / num_cores)): int(
                (cnt) * (len(tuples) / num_cores)) - 1]
            p = Process(target=generate_feat_engineered_feats, args=(dataset, partial_tuples, num_feat_to_select_and_comb, my_queue, opr_list, num_cores))
            T.append(p)
        for t in T[(i - 1) * num_procs: i * num_procs - 1]:
            t.start()
        for t in T[(i - 1) * num_procs: i * num_procs - 1]:
            t.join()

        for t in T[(i - 1) * num_procs: i * num_procs - 1]:
            x = my_queue.get()
            temp_dataset = pd.concat([temp_dataset, x],axis=1)

        # list_of_feat_to_rmv.append([x for x in my_queue.get()])

    dataset = pd.concat([dataset, temp_dataset], axis=1)
    dataset = dataset.loc[:, ~dataset.columns.duplicated()]
    return dataset







def feature_engineering(dataset, num_feat_to_select_and_comb=2,
                        opr_list=['Add', 'Sub', 'Mult', 'Div', 'Sin', 'Cos', 'Exp'], num_cores = 8):
    # this module receives a dataset and combines features to engineer more features.
    # if the num_feat_to_select_and_comb = None, then it automatically combines the features that are the most correlated (based on pierson correlation);
    # otherwise it selects num_feat_to_select_and_comb from all the features and make all the combinations based on these methods:
    # 1) multiplication
    # 2) Log-Multi
    new_dataset = pd.DataFrame([])
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


    if (num_feat_to_select_and_comb == 2):
        for tuple in tuples:
            if "Add" in opr_list:
                new_add_label = str(tuple[0]) + str('_Add_') + str(tuple[1])
                new_add_data = np.add(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values))
                dataset[new_add_label] = new_add_data
            if "Sub" in opr_list:
                new_subt_label = str(tuple[0]) + str('_Sub_') + str(tuple[1])
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

    if (num_feat_to_select_and_comb == 3):
        for tuple in tuples:
            if "Add" in opr_list:
                new_add_label = str(tuple[0]) + str('_Add_') + str(tuple[1]) + str('_Add_') + str(tuple[2])
                new_add_label = str(tuple[0]) + str('_Add_') + str(tuple[1]) + str('_Add_') + str(tuple[2])
                new_add_data = np.add(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                      np.array(dataset[tuple[2]].values))
                dataset[new_add_label] = new_add_data
            if "Sub" in opr_list:
                new_subt_label = str(tuple[0]) + str('_Sub_') + str(tuple[1])
                new_subt_data = np.subtract(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                            np.array(dataset[tuple[2]].values))
                dataset[new_subt_label] = new_subt_data

            if "Mult" in opr_list:
                new_Mult_label = str(tuple[0]) + str('_Mult_') + str(tuple[1])
                new_Mult_data = np.multiply(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                            np.array(dataset[tuple[2]].values))
                dataset[new_Mult_label] = new_Mult_data

            if "Div" in opr_list:
                new_Div_label = str(tuple[0]) + str('_Div_') + str(tuple[1])
                new_Div_data = np.divide(np.array(dataset[tuple[0]].values), np.array(dataset[tuple[1]].values),
                                         np.array(dataset[tuple[2]].values))
                dataset[new_Div_label] = new_Div_data

    dataset_with_new_comb_feats = dataset

    return dataset_with_new_comb_feats


# Python code to remove duplicate elements
def remove_duplicates(input_list):
    final_list = []
    for num in input_list:
        if num not in final_list:
            final_list.append(num)
    return final_list


def drop_cocorr(df, corr_thr):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > corr_thr)]

    # Drop features
    d = df.drop(df[to_drop], axis=1)

    return d


def find_highly_corr_feat(dataset, partial_list,corr_thr, my_queue):
    df = dataset[partial_list]


    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > corr_thr)]

    #
    # to_drop = [column for column in upper.columns if any(upper[column] > corr_thr)]
    #
    # list_of_feat_to_rmv = []
    # for tuple in partial_tuple:
    #
    #     if (abs(dataset[list(tuple)].corr('pearson')[tuple[1]][0]) >= corr_thr):
    #         list_of_feat_to_rmv.append(tuple[0])


    #my_queue.put(list_of_feat_to_rmv)
    my_queue.put(to_drop)

def remove_corr_feats(dataset, corr_thr, num_cores=8, num_procs=100):
    m = multiprocessing.Manager()
    my_queue = m.Queue()


    T = []
    list_of_feat_to_rmv = []
    for i in range(1, int((num_cores ) / num_procs)+1):
        for j in range(1, num_procs+1):
            cnt = (i - 1) * num_procs + j

            partial_list = dataset.columns.values[int((cnt - 1) * (len(dataset.columns.values) / num_cores)): int((cnt) * (len(dataset.columns.values) / num_cores)) - 1]
            p = Process(target=find_highly_corr_feat, args=(dataset, partial_list, corr_thr, my_queue))
            T.append(p)
        for t in T[(i - 1) * num_procs: i * num_procs - 1]:
            t.start()
        for t in T[(i - 1) * num_procs: i * num_procs - 1]:
            t.join()

        for t in T[(i - 1) * num_procs: i * num_procs - 1]:
            for x in my_queue.get():
                list_of_feat_to_rmv.append(x)



        # list_of_feat_to_rmv.append([x for x in my_queue.get()])

    dataset = dataset.drop(remove_duplicates(list_of_feat_to_rmv), axis=1)
    return dataset



    # # this module consider every two features, measures their ‘pearson’ correlation and remove one of them
    # # if their corr is above "corr_thr"
    # tuples = list(combinations(dataset.columns, 2))
    # T = []
    # for i in range(1, num_cores+1):
    #     partial_tuple = tuples[int((i-1)*(len(tuples)/num_cores)): int((i)*(len(tuples)/num_cores)) - 1]
    #     p = Process(target=find_highly_corr_feat, args=(dataset, partial_tuple,corr_thr,my_queue))
    #     T.append(p)
    # for t in T:
    #     t.start()
    # for t in T:
    #     t.join()
    #
    # list_of_feat_to_rmv = []
    # for t in T:
    #     for x in my_queue.get():
    #         list_of_feat_to_rmv.append(x)
    #
    #     #list_of_feat_to_rmv.append([x for x in my_queue.get()])
    #
    #
    # dataset = dataset.drop(remove_duplicates(list_of_feat_to_rmv), axis=1)
    # return dataset


def odin(dataset, num_cores, target_feature='next_week_yield', export_corr_hist='True', num_of_most_corr_feats=10,
         feature_engineering=False,
         feature_select_methods=['Most-Corr', 'UniVar-Select', 'Rec-Feat-Elimin', 'Prin-Comp-Anal', 'Feat-Import'],
         opr_list=["Mult"],
         num_procs = 100
         ):
    # This module receives a Dataset and a target_feature
    # 1) it produces ABS-corr histogram against the target_feature.
    # 2) it return the num_of_most_corr featuresf
    # 3) it generates a histogram file if export_corr_hist = True.
    # 4) if feature_engineering is true,
    #    4-1) it generates different combinations of columns
    #dataset = dataset.dropna(axis=1)
    if (dataset[target_feature].isnull().values.any()):
        raise ValueError('There is a NaN in the target column. We cannot fill out those NaN using average.')
    dataset = dataset.fillna(dataset.mean())  # fill out any NaN in feature columns.
    # dataset = dataset.dropna()
    dataset = dataset.astype(float)

    if (export_corr_hist == True):
        dataVisualization(data=dataset, target_feat=target_feature)
        print("\n Data visualization has ended. \n")

    most_corr_sorted_labels = feature_selection(dataset=dataset, num_cores=int(num_cores * len(dataset)/40), target_feature=target_feature,
                                                feature_select_method=feature_select_methods, opr_list=opr_list, num_procs = num_procs)
    print("most corr features are: %s \n" % (most_corr_sorted_labels))

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
        ensemble_size=3,
        initial_configurations_via_metalearning=0,
        # include_preprocessors=["no_preprocessing"],  # in a case that we do not autoML does any feature-engineering
        time_left_for_this_task=processing_time_in_secs,
        per_run_time_limit=int(processing_time_in_secs / 10),
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
        seed=7,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        shared_mode=True,
        ensemble_memory_limit=4096,
        ml_memory_limit=10000
    )

    automl.fit(train_X.copy(),
               train_y.copy(),
               dataset_name='dataset',
               feat_type=feature_types,
               metric=autosklearn.metrics.mean_squared_error
               )
    automl.refit(train_X.copy(), train_y.copy())

    print(automl.show_models())
    print(automl.sprint_statistics())

    return automl


def h2o_design_space_search(train_X_labels, train_y_label, train_dataset, test_X, max_runtime_secs):
    seed = 7
    np.random.seed(seed)
    h2o.init(max_mem_size_GB=3)

    # Identify predictors and response
    x = train_X_labels
    y = train_y_label

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

    # If you need to generate predictions on a test set, you can make
    # predictions directly on the `"H2OAutoML"` object, or on the leader
    # model object directly

    test_X = h2o.H2OFrame(test_X)
    # preds = aml.predict(test_X)

    # or:
    preds = aml.leader.predict(test_X)
    preds = [float(x) for x in
             preds.get_frame_data().split(sep='\n')[1:len(preds.get_frame_data().split(sep='\n')) - 1]]
    # preds = np.array(preds) - (0.9 * lb[1, 5]) # minuse the yield values from their MAE
    print("predicted values before manipulating them in rising edges:" + str(preds))
    new_preds = preds
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


def roll_accum_and_add_new_feat(dataset, num_roll_and_accum):
    new_dataset = pd.DataFrame([])
    for i in range(2, num_roll_and_accum):
        new_dataset = pd.concat([new_dataset, dataset.rolling(i, min_periods=1).sum().add_suffix("_seas_roll_" + str(i))],
                                axis=1)
    return new_dataset




def select_to_do_prediction_for_how_many_weeks_in_advance(dataset, num_cores):
    temp_dataset = dataset.copy()
    yield_related_feat_list = [ 'y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']
    yield_related_feat = temp_dataset[['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']]
    temp_dataset = temp_dataset.drop([ 'y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected'], axis=1)
    #rolled_dataset = roll_accum_and_add_new_feat(temp_dataset, 5)
    temp_dataset[['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']] = yield_related_feat
    rolled_dataset = pd.DataFrame([], temp_dataset.index)
    # mov_win_dataset = mov_win_and_add_new_feat(temp_dataset,5)
    temp_dataset = pd.concat([temp_dataset, rolled_dataset], axis=1)
    temp_temp_dataset = temp_dataset.copy()
    max_corr = -1
    for i in range(1, 9):
        temp_dataset = temp_temp_dataset.copy()
        target_feature = 'y' + str(i)
        temp_dataset = temp_dataset[np.isfinite(temp_dataset[target_feature])]
        temp_dataset = temp_dataset[(temp_dataset[[target_feature]] != 0).all(axis=1)]
        temp_dataset = temp_dataset[:-1 * i]

        yield_related_feat_list.remove(target_feature)
        # pd.concat([temp_dataset, yield_related_feat['y' + str(i)]], axis=1)
        list_of_most_corr_feat = feature_selection(dataset=temp_dataset.drop(yield_related_feat_list, axis=1), num_cores=num_cores,
                                                   target_feature=target_feature, feature_select_method='Most-Corr',
                                                   opr_list="[]")
        yield_related_feat_list.append(target_feature)
        corr_dataframe = pd.DataFrame([], index=['corr'])
        try:
            for clmn_label in list_of_most_corr_feat[0:2]:
                calc_corr_dataset = pd.DataFrame([])
                calc_corr_dataset[clmn_label] = temp_dataset[clmn_label]
                calc_corr_dataset[target_feature] = yield_related_feat[target_feature]
                try:
                    corr_dataframe[clmn_label] = pd.Series(
                        abs(calc_corr_dataset[calc_corr_dataset.columns].corr()[target_feature][:-1]).iloc[0],
                        corr_dataframe.index)
                    print("corr between %s and %s is: %s" % (
                    target_feature, clmn_label, corr_dataframe[clmn_label].iloc[0]))
                except IndexError:
                    continue
            if (corr_dataframe.values.mean() > max_corr):
                max_corr = corr_dataframe.values.mean()
                label = target_feature
        except KeyError:
            print("ERROR: Probably there is a nan or a VALUE error in your features columns.")
            raise
    return label


def bishop(dataset, target_feat):
    target_feat_values = dataset[target_feat]
    for target_feat_index in target_feat_values.index:
        print(target_feat_index)



def convert_numbers_with_comma(dataset):
    """
    Convert the string number value to a float
     - Remove $
     - Remove commas
     - Convert to float type
    """
    for clmn in dataset.columns:
        dataset[clmn] = dataset[clmn].replace(',','')

    return dataset


def roll_seasanal_and_add_feat(dataset, start_and_end_of_seas, rolling_range):

    rolled_dataset = pd.DataFrame([])
    for (start, end) in start_and_end_of_seas:
        seas_dataset = dataset.loc[start: end]
        rolled_seas_dataset = roll_accum_and_add_new_feat(seas_dataset, rolling_range)
        rolled_dataset = pd.concat([rolled_dataset, rolled_seas_dataset], axis = 0)

    return rolled_dataset


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

    if args.rmv_feat_with_corr_higher_than:
        rmv_feat_with_corr_higher_than = args.rmv_feat_with_corr_higher_than
    else:
        rmv_feat_with_corr_higher_than = config.rmv_feat_with_corr_higher_than

    if args.slope:
        slope = args.slope
    else:
        slope = config.slope

    if args.bias:
        bias = args.bias
    else:
        bias = config.bias

    if args.target_feat:
        target_feat = args.target_feat
    else:
        target_feat = config.target_feat

    if args.drop_feat:
        drop_feat = args.drop_feat
    else:
        drop_feat = config.drop_feat

    if args.num_week_shift:
        num_week_shift = args.num_week_shift
    else:
        num_week_shift = config.num_week_shift

    if args.alr_gen_feats != None:
        alr_gen_feats = args.alr_gen_feats
    else:
        alr_gen_feats = config.alr_gen_feats

    if (args.add_mov_win_rolling_feats != None):
        add_mov_win_rolling_feats = args.add_mov_win_rolling_feats
    else:
        add_mov_win_rolling_feats = config.add_mov_win_rolling_feats

    if (args.match_with_actual_yield != None):
        match_with_actual_yield = args.match_with_actual_yield
    else:
        match_with_actual_yield = config.match_with_actual_yield

    if (args.intel_feat_moving != None):
        intel_feat_moving = args.intel_feat_moving
    else:
        intel_feat_moving = config.intel_feat_moving

    if (args.num_procs != None):
        num_procs = args.num_procs
    else:
        num_procs = config.num_procs


    num_cores = 15 * psutil.cpu_count()  # automatically set the number of used cores based on the number of logical cores.

    if (alr_gen_feats == False):
        # load dataset
        if (verbose > 0):
            print("###########################################")
            print("datset is read from" + str(
                dataset_file_add) + "." + "\n make sure that the format within file is followed correctly.")
            print("###########################################")
        # dataset = pd.read_excel(dataset_file_add, sheetname='entire_data')
        dataset = pd.read_csv(dataset_file_add)

        if (verbose > 5):
            print(dataset)
        elif (verbose > 0 and verbose <= 5):
            print("dataset column names are:", str(dataset.columns.values))

        # based on the input file format there should be two first columns in the dataset_file that are translated into indeces (refer to the example input file).

        dataset = dataset.set_index(["iso_date"]) #dataset.set_index(['year',	'week_gw']) #dataset.set_index(["date", "iso_date"]) #dataset.set_index(['year',	'week_gw'])#dataset.set_index(["date", "iso_date"])  # dataset.set_index(['week_gw']) #
        dataset = dataset.drop(['date'], axis = 1)

        try:
            dataset = dataset.drop(labels=drop_feat, axis=1)
        except KeyError:
            print("there was no label with %s name to drop!" % (drop_feat))
            # getch.getch()

        # remove constant columns
        #dataset = dataset.loc[:, (dataset != dataset.iloc[0]).any()]
        #dataset = dataset.loc[:, dataset.apply(pd.Series.nunique) != 1]

        for clmn in dataset.columns:
            cnt = 0
            for cell in dataset[clmn]:
                try:
                    float(cell)
                except ValueError:
                    dataset[clmn][cnt] = np.nan
                cnt += 1

            # remove columns (features) that are more than 1/3 empty.
        # remove features that are too sparse more than 1/3 is empty
        s = (dataset.isna().sum() < ((4 / 6) * dataset.shape[0]))
        print(s)
        dataset = dataset.iloc[:, s.values]
        print(dataset.columns.values)

        #dataset = convert_numbers_with_comma(dataset)


        # make sure that the type of all entries in your dataset are float not string or boolean since they are used in ML training.
        dataset = dataset.astype(float)

        projected_yield = dataset['projected']
        # target_feat = dataset[target_feat]
        prev_week_yield = dataset['actual']

        target_related_feats = dataset[[ 'y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']]

        dataset_without_yield_related_params = dataset.drop([ 'y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected'], axis=1)
        dataset_without_yield_related_params = remove_corr_feats(dataset_without_yield_related_params,
                                                                 rmv_feat_with_corr_higher_than,
                                                                 num_procs = num_procs)
        #########################################################
        ######################## Rimato #########################
        #########################################################

        # dataset_without_yield_related_params_rolled = roll_seasanal_and_add_feat(dataset_without_yield_related_params,
        #                                                                          rolling_range=8,
        #                                                                          start_and_end_of_seas=[
        #                                                                              ("Week 40 FY15", "Week 37 FY16"),
        #                                                                              ("Week 38 FY16", "Week 32 FY17"),
        #                                                                              ("Week 33 FY17", "Week 27 FY18"),
        #                                                                              ("Week 28 FY18", "Week 20 FY19")])
        # dataset = pd.concat([dataset, dataset_without_yield_related_params_rolled], axis=1, sort=False)
        # dataset = dataset.drop(
        #     ["Week 40 FY15", "Week 41 FY15", "Week 42 FY15", "Week 43 FY15", "Week 44 FY15", "Week 45 FY15",
        #      "Week 46 FY15", "Week 47 FY15", "Week 48 FY15", "Week 30 FY16", "Week 31 FY16", "Week 32 FY16",
        #      "Week 33 FY16", "Week 34 FY16", "Week 35 FY16", "Week 36 FY16", "Week 37 FY16", "Week 38 FY16",
        #      "Week 39 FY16", "Week 40 FY16", "Week 41 FY16", "Week 42 FY16", "Week 43 FY16", "Week 44 FY16",
        #      "Week 45 FY16", "Week 46 FY16", "Week 47 FY16", "Week 48 FY16", "Week 24 FY17", "Week 25 FY17",
        #      "Week 26 FY17", "Week 27 FY17", "Week 28 FY17", "Week 29 FY17", "Week 30 FY17", "Week 31 FY17",
        #      "Week 32 FY17", "Week 33 FY17", "Week 34 FY17", "Week 35 FY17", "Week 36 FY17", "Week 37 FY17",
        #      "Week 38 FY17", "Week 39 FY17", "Week 40 FY17", "Week 41 FY17", "Week 19 FY18", "Week 20 FY18",
        #      "Week 21 FY18", "Week 22 FY18", "Week 23 FY18", "Week 24 FY18", "Week 25 FY18", "Week 26 FY18",
        #      "Week 27 FY18", "Week 28 FY18", "Week 29 FY18", "Week 30 FY18", "Week 31 FY18", "Week 32 FY18",
        #      "Week 33 FY18", "Week 34 FY18", "Week 35 FY18", "Week 36 FY18"], axis=0)
        #########################################################


        ###################### Agrimind ##############
        # dataset_without_yield_related_params_rolled = roll_seasanal_and_add_feat(dataset_without_yield_related_params, rolling_range=8, start_and_end_of_seas = [("Week 1 FY15", "Week 51 FY15"), ("Week 52 FY15", "Week 49 FY16"), ("Week 50 FY16", "Week 51 FY17"), ("Week 52 FY17", "Week 34 FY18"), ("Week 35 FY18", "Week 46 FY18"), ("Week 47 FY18", "Week 20 FY19")])
        # dataset = pd.concat([dataset, dataset_without_yield_related_params_rolled], axis =1, sort=False)
        # dataset = dataset.drop([ "Week 1 FY15", "Week 2 FY15", "Week 3 FY15", "Week 4 FY15", "Week 5 FY15", "Week 6 FY15", "Week 7 FY15", "Week 8 FY15", "Week 9 FY15", "Week 10 FY15", "Week 11 FY15", "Week 12 FY15", "Week 13 FY15", "Week 14 FY15", "Week 15 FY15", "Week 16 FY15", "Week 17 FY15", "Week 18 FY15", "Week 19 FY15", "Week 20 FY15", "Week 21 FY15", "Week 22 FY15", "Week 23 FY15", "Week 24 FY15", "Week 25 FY15", "Week 26 FY15", "Week 27 FY15", "Week 28 FY15", "Week 29 FY15", "Week 30 FY15", "Week 31 FY15", "Week 32 FY15", "Week 33 FY15", "Week 34 FY15", "Week 35 FY15", "Week 36 FY15", "Week 37 FY15", "Week 38 FY15", "Week 39 FY15", "Week 40 FY15", "Week 41 FY15", "Week 42 FY15", "Week 43 FY15", "Week 44 FY15", "Week 45 FY15", "Week 46 FY15", "Week 47 FY15", "Week 48 FY15", "Week 49 FY15", "Week 50 FY15", "Week 51 FY15", "Week 52 FY15", "Week 53 FY15", "Week 1 FY16", "Week 2 FY16", "Week 3 FY16", "Week 4 FY16", "Week 5 FY16", "Week 6 FY16", "Week 41 FY16", "Week 42 FY16", "Week 43 FY16", "Week 44 FY16", "Week 45 FY16", "Week 46 FY16", "Week 47 FY16", "Week 48 FY16", "Week 49 FY16", "Week 50 FY16", "Week 51 FY16", "Week 52 FY16", "Week 1 FY17", "Week 2 FY17", "Week 3 FY17", "Week 4 FY17", "Week 5 FY17", "Week 6 FY17", "Week 7 FY17", "Week 8 FY17", "Week 9 FY17", "Week 43 FY17", "Week 44 FY17", "Week 45 FY17", "Week 46 FY17", "Week 47 FY17", "Week 48 FY17", "Week 49 FY17", "Week 50 FY17", "Week 51 FY17", "Week 52 FY17", "Week 1 FY18", "Week 2 FY18", "Week 3 FY18", "Week 37 FY18", "Week 38 FY18", "Week 39 FY18", "Week 40 FY18", "Week 41 FY18", "Week 42 FY18", "Week 43 FY18", "Week 44 FY18", "Week 45 FY18", "Week 46 FY18", "Week 47 FY18", "Week 48 FY18", "Week 49 FY18", "Week 50 FY18", "Week 51 FY18", "Week 52 FY18" ], axis=0)
        ############################################

        ################### SERA ####################
        dataset_without_yield_related_params_rolled = roll_seasanal_and_add_feat(dataset_without_yield_related_params,
                                                                                 rolling_range=8,
                                                                                 start_and_end_of_seas=[
                                                                                     ("Week 53 FY15", "Week 29 FY16"),
                                                                                     ("Week 30 FY16", "Week 29 FY17"),
                                                                                     ("Week 30 FY17", "Week 28 FY18"),
                                                                                     ("Week 29 FY18", "Week 19 FY19")])
        dataset = pd.concat([dataset, dataset_without_yield_related_params_rolled], axis=1, sort=False)
        dataset = dataset.drop(
            ["Week 53 FY15", "Week 1 FY16", "Week 21 FY16", "Week 22 FY16", "Week 23 FY16", "Week 24 FY16",
             "Week 25 FY16", "Week 26 FY16", "Week 27 FY16", "Week 28 FY16", "Week 29 FY16", "Week 30 FY16",
             "Week 31 FY16", "Week 32 FY16", "Week 33 FY16", "Week 34 FY16", "Week 35 FY16", "Week 36 FY16",
             "Week 37 FY16", "Week 38 FY16", "Week 39 FY16", "Week 40 FY16", "Week 21 FY17", "Week 22 FY17",
             "Week 23 FY17", "Week 24 FY17", "Week 25 FY17", "Week 26 FY17", "Week 27 FY17", "Week 28 FY17",
             "Week 29 FY17", "Week 30 FY17", "Week 31 FY17", "Week 32 FY17", "Week 33 FY17", "Week 34 FY17",
             "Week 35 FY17", "Week 36 FY17", "Week 37 FY17", "Week 38 FY17", "Week 39 FY17", "Week 20 FY18",
             "Week 21 FY18", "Week 22 FY18", "Week 23 FY18", "Week 24 FY18", "Week 25 FY18", "Week 26 FY18",
             "Week 27 FY18", "Week 28 FY18", "Week 29 FY18", "Week 30 FY18", "Week 31 FY18", "Week 32 FY18",
             "Week 33 FY18", "Week 34 FY18", "Week 35 FY18", "Week 36 FY18", "Week 37 FY18", "Week 38 FY18",
             "Week 39 FY18"], axis=0)
        ###############################################

        ################### SanLucar #################
        # dataset_without_yield_related_params_rolled = roll_seasanal_and_add_feat(dataset_without_yield_related_params,
        #                                                                          rolling_range=5,
        #                                                                          start_and_end_of_seas=[
        #                                                                              ("Week 36 FY14", "Week 23 FY15"),
        #                                                                              ("Week 24 FY15", "Week 20 FY16"),
        #                                                                              ("Week 21 FY16", "Week 21 FY17"),
        #                                                                              ("Week 22 FY17", "Week 20 FY18"),
        #                                                                              ("Week 21 FY18", "Week 18 FY19")])
        # dataset = pd.concat([dataset, dataset_without_yield_related_params_rolled], axis=1, sort=False)
        # dataset = dataset.drop(
        #     ["Week 36 FY14", "Week 37 FY14", "Week 38 FY14", "Week 39 FY14", "Week 40 FY14", "Week 41 FY14",
        #      "Week 42 FY14", "Week 43 FY14", "Week 44 FY14", "Week 45 FY14", "Week 15 FY15", "Week 16 FY15",
        #      "Week 17 FY15", "Week 18 FY15", "Week 19 FY15", "Week 20 FY15", "Week 21 FY15", "Week 22 FY15",
        #      "Week 23 FY15", "Week 24 FY15", "Week 25 FY15", "Week 26 FY15", "Week 36 FY15", "Week 37 FY15",
        #      "Week 38 FY15", "Week 39 FY15", "Week 40 FY15", "Week 41 FY15", "Week 42 FY15", "Week 43 FY15",
        #      "Week 44 FY15", "Week 45 FY15", "Week 46 FY15", "Week 47 FY15", "Week 13 FY16", "Week 14 FY16",
        #      "Week 15 FY16", "Week 16 FY16", "Week 17 FY16", "Week 18 FY16", "Week 19 FY16", "Week 20 FY16",
        #      "Week 21 FY16", "Week 22 FY16", "Week 23 FY16", "Week 24 FY16", "Week 25 FY16", "Week 26 FY16",
        #      "Week 36 FY16", "Week 37 FY16", "Week 38 FY16", "Week 39 FY16", "Week 40 FY16", "Week 41 FY16",
        #      "Week 42 FY16", "Week 43 FY16", "Week 44 FY16", "Week 45 FY16", "Week 46 FY16", "Week 13 FY17",
        #      "Week 14 FY17", "Week 15 FY17", "Week 16 FY17", "Week 17 FY17", "Week 18 FY17", "Week 19 FY17",
        #      "Week 20 FY17", "Week 21 FY17", "Week 22 FY17", "Week 23 FY17", "Week 24 FY17", "Week 25 FY17",
        #      "Week 38 FY17", "Week 39 FY17", "Week 40 FY17", "Week 41 FY17", "Week 42 FY17", "Week 43 FY17",
        #      "Week 44 FY17", "Week 45 FY17", "Week 46 FY17", "Week 47 FY17", "Week 13 FY18", "Week 14 FY18",
        #      "Week 15 FY18", "Week 16 FY18", "Week 17 FY18", "Week 18 FY18", "Week 19 FY18", "Week 20 FY18",
        #      "Week 21 FY18", "Week 22 FY18", "Week 23 FY18", "Week 24 FY18", "Week 37 FY18", "Week 38 FY18",
        #      "Week 39 FY18", "Week 40 FY18", "Week 41 FY18", "Week 42 FY18", "Week 43 FY18", "Week 44 FY18",
        #      "Week 45 FY18", "Week 46 FY18", "Week 47 FY18"], axis=0)
        ##############################################


        #dataset.to_csv('../Rimato/rolled_dataset.csv')

        dataset = remove_corr_feats(dataset, rmv_feat_with_corr_higher_than)

        #dataset = drop_cocorr(dataset.drop([ 'y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected'], axis=1),
        #                      rmv_feat_with_corr_higher_than)

        dataset[['y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']] = target_related_feats
        dataset['projected'] = projected_yield
        #

        which_week_to_predict = select_to_do_prediction_for_how_many_weeks_in_advance(dataset, num_cores)

        which_week_to_predict = target_feat

        predict_yield_labels = ['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1']
        predict_yield_labels.remove(which_week_to_predict)

        target_feat = which_week_to_predict
        actual_yield = dataset[target_feat]

        list_of_target_feat_to_remove = ['y' + str(x) for x in range(1, 9)]
        list_of_target_feat_to_remove.remove(target_feat)
        target_related_feats = dataset[['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']]

        dataset = dataset.drop(list_of_target_feat_to_remove +['actual', 'projected'], axis=1)

        # rolled_dataset = roll_accum_and_add_new_feat(dataset, 9)
        # dataset = pd.concat([dataset, rolled_dataset], axis=1)
        # dataset = drop_cocorr(dataset.drop([], axis=1), rmv_feat_with_corr_higher_than)


        dataset[['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']] = target_related_feats
        dataset = dataset.drop(['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1'], axis=1)

        #dataset = pd.DataFrame(SoftImpute().fit_transform(dataset), index=dataset.index, columns=dataset.columns.values)

        # get a copy of the 'actual_yield' and 'projected_yield' before normalization (standardization).
        # we do not want to Norm target parameters.

        #dataset = drop_cocorr(dataset.drop(['projected'], axis=1), rmv_feat_with_corr_higher_than)
        dataset = remove_corr_feats(dataset.drop(['projected'], axis=1), rmv_feat_with_corr_higher_than)
        #dataset [target_feat] = actual_yield
        #dataset = dataset.drop(['actual','y6'], axis =1) # we cannot roll actual yield or any data directrly related to yield values. Otherwise we are cheating!
        #dataset = roll_accum_and_add_new_feat(dataset, 4)

        dataset = drop_cocorr(dataset.drop([], axis=1), rmv_feat_with_corr_higher_than)
        dataset[target_feat] = actual_yield
        try:
            dataset = dataset.drop(['actual'],axis=1)
        except KeyError:
            dataset = dataset



        dataset = move_feats_and_add_new_feats(dataset, max_num_mov= (-1 * int(target_feat[1]))+2, min_num_mov= (-1 * int(target_feat[1])) - 1,
                                               target_feat=target_feat, intel_feat_moving=intel_feat_moving)

        #rolled_dataset = roll_accum_and_add_new_feat(dataset, 9)
        #dataset = pd.concat([dataset, rolled_dataset], axis=1)
        #dataset = drop_cocorr(dataset.drop([], axis=1), rmv_feat_with_corr_higher_than)

        #dataset = dataset.drop(target_feat, axis =1) # we cannot roll actual yield or any data directrly related to yield values. Otherwise we are cheating!
        #dataset = roll_accum_and_add_new_feat(dataset, 4)


        #dataset = drop_cocorr(dataset, rmv_feat_with_corr_higher_than)
        dataset = remove_corr_feats(dataset, rmv_feat_with_corr_higher_than)

        dataset[['y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']] = target_related_feats
        which_week_to_predict = select_to_do_prediction_for_how_many_weeks_in_advance(dataset, num_cores)


        which_week_to_predict = target_feat
        dataset = dataset.drop(['y8', 'y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1'], axis=1)

        dataset['actual'] = prev_week_yield

        #dataset = drop_cocorr(dataset.drop([], axis=1), rmv_feat_with_corr_higher_than)
        dataset = remove_corr_feats(dataset.drop([], axis=1), rmv_feat_with_corr_higher_than)

        dataset[target_feat] = actual_yield

        dataset = dataset[np.isfinite(dataset[target_feat])]
        dataset = dataset[(dataset[[target_feat]] != 0).all(axis=1)]
        actual_yield = dataset[target_feat]

        dataset = dataset.fillna(0)
        #dataset = pd.DataFrame(SoftImpute().fit_transform(dataset), index=dataset.index, columns=dataset.columns.values)

        if (verbose > 0):
            print("#############################")
            print("END OF REMOVE CORR FEATURES!")
            print("#############################")

        dataset['actual'] = prev_week_yield

        dataset[['y8','y7', 'y6', 'y5', 'y4', 'y3', 'y2', 'y1', 'actual', 'projected']] = target_related_feats

        dataset = standardization(dataset)

        try:
            dataset = dataset.drop([target_feat], axis=1)
        except KeyError:
            dataset = dataset

        try:
            dataset = dataset.drop('projected', axis=1)
        except KeyError:
            dataset = dataset

        try:
            dataset = dataset.drop(list_of_target_feat_to_remove, axis=1)
        except KeyError:
            dataset = dataset

        #copy_dataset = dataset.copy()
        # do feature engineering on the input dataset
        # remove actual_yield and projected_yield from the features that we do feat_eng on them.
        # we shouldn't rely on 'projected_yield' as an input feat to our yield prediction models.
        feat_engineered_dataset = feature_engineering_parallely(
            dataset=dataset,
            num_feat_to_select_and_comb=2,
            opr_list=oprs_for_feat_eng,
            num_cores=num_cores,
            num_procs = num_procs
        )

        # [feat_engineered_dataset, list_of_feat_to_rmv] = remove_corr_feats(feat_engineered_dataset.drop([], axis=1), rmv_feat_with_corr_higher_than)

        if (verbose > 0):
            print("#############################")
            print("END OF FEATURES ENGINEERING!")
            print("#############################")
        # receive feature engineered dataset which is standardized and replace standardized "actual_yield" with its real values
        # note that we do not want to predict standardized target values ("actual_yield"). We want to measure MAE and RMSE which requires real values for "actual_yield"
        feat_engineered_dataset[target_feat] = actual_yield.values

        # select the "num_of_most_corr_feats" most correlated features based on "feature_select_methods" from the feature engineered dataset.
        # also odin save the correlation map onto a file.
        if (verbose > 0):
            print("################################")
            print("Data Statis Visualization and feature selection have started....")
        print("data visul flag ", dataset_stat_visual_flag)





        name_of_most_corr_feat = odin(dataset=feat_engineered_dataset, num_cores=num_cores, target_feature=target_feat,
                                      export_corr_hist=dataset_stat_visual_flag,
                                      num_of_most_corr_feats=num_of_most_corr_feats,
                                      feature_select_methods=feature_select_methods, opr_list=oprs_for_feat_eng,
                                      num_procs = num_procs)

        try:
            for feat_name in name_of_most_corr_feat:
                if target_feat in feat_name:
                    name_of_most_corr_feat.remove(feat_name)

                if 'actual' in feat_name:
                    name_of_most_corr_feat.remove(feat_name)
        except ValueError:
            name_of_most_corr_feat = name_of_most_corr_feat

        list_of_impr_feat_for_H2O = name_of_most_corr_feat  # we exclude actual_yield from the feature's list

        if (verbose > 0):
            print("the labels for the most corr features are:", str(list_of_impr_feat_for_H2O))
        for most_corr_feat_label in list_of_impr_feat_for_H2O:
            print("Pearson Corr of %s to %s is %s" % (target_feat, most_corr_feat_label, abs(
                feat_engineered_dataset[[target_feat, most_corr_feat_label]].corr()).iloc[0, 1]))

        dataset = feat_engineered_dataset[name_of_most_corr_feat]
        del feat_engineered_dataset  # to release memory

        # dataset_with_new_comb_feats = standardization(dataset_with_new_comb_feats)

        # dataset = dataset.drop(['fed_yield'],axis=1)
        # train_dataset = dataset[0:int(train_precental * len(dataset))]
        # test_dataset = dataset[int(train_precental * len(dataset)):]

        # listOfTupleNames = list((list_of_impr_feat_for_H2O, len(list_of_impr_feat_for_H2O)))
        features_and_their_MAE = 30

        # dataVisualization(data=dataset)

        # dataset = dataset.drop(['projected_yield'],axis=1)

        dataset[target_feat] = actual_yield
        # train_dataset = dataset[0:int(train_precental * len(dataset))]

        dataset['projected'] = projected_yield
        # dataset['actual'] = prev_week_yield
        dataset.to_pickle("./feat_eng_featurs.csv")
        #dataset['projected'] = [1 for x in range(0, len(projected_yield))]
        #projected_yield = dataset['projected']
        list_of_impr_feat_for_H2O = list(dataset.columns.values)
    else:
        dataset = pd.read_pickle("./feat_eng_featurs.csv")
        if (add_mov_win_rolling_feats == True):
            rolled_dataset = roll_accum_and_add_new_feat(dataset.drop([target_feat, 'projected'], axis = 1), 6)
            dataset = pd.concat([dataset, rolled_dataset], axis=1)

        name_of_most_corr_feat = odin(dataset=dataset, num_cores=num_cores, target_feature=target_feat,
                                      export_corr_hist=dataset_stat_visual_flag,
                                      num_of_most_corr_feats=num_of_most_corr_feats,
                                      feature_select_methods=feature_select_methods, opr_list=oprs_for_feat_eng,
                                      num_procs = num_procs)


        list_of_impr_feat_for_H2O = name_of_most_corr_feat
        actual_yield = dataset[target_feat]
        projected_yield = dataset['projected']
        # prev_week_yield = dataset['actual']
        dataset['projected'] = [0.1 for x in range(0, len(projected_yield)) ]
        projected_yield = dataset['projected']
    try:
        for feat_name in list_of_impr_feat_for_H2O:
            if target_feat in feat_name:
                list_of_impr_feat_for_H2O.remove(feat_name)

            if 'actual' in feat_name:
                list_of_impr_feat_for_H2O.remove(feat_name)
            if 'projected' in feat_name:
                list_of_impr_feat_for_H2O.remove(feat_name)
    except ValueError:
        list_of_impr_feat_for_H2O = list_of_impr_feat_for_H2O

    try:
        list_of_impr_feat_for_H2O.remove('projected')
    except ValueError:
        list_of_impr_feat_for_H2O = list_of_impr_feat_for_H2O
    try:
        list_of_impr_feat_for_H2O.remove('actual')
    except ValueError:
        list_of_impr_feat_for_H2O = list_of_impr_feat_for_H2O
    try:
        list_of_impr_feat_for_H2O.remove(target_feat)
    except ValueError:
        list_of_impr_feat_for_H2O = list_of_impr_feat_for_H2O

    list_of_impr_feat_for_H2O = list_of_impr_feat_for_H2O[0:num_of_most_corr_feats]

    dataset['projected'] = projected_yield



    train_dataset = dataset[0:math.floor(int(train_precental * len(dataset)))]
    train_dataset = train_dataset[list_of_impr_feat_for_H2O + [target_feat]]

    test_dataset = dataset[math.floor(int(train_precental * len(dataset))) + 1:]
    test_manual_preds = list(test_dataset['projected'][1:len(test_dataset['projected'])]) + [
        test_dataset['projected'][1]]

    test_dataset = test_dataset[list_of_impr_feat_for_H2O + [target_feat]]

    test_X = test_dataset[list(list_of_impr_feat_for_H2O)]

    test_y = test_dataset[target_feat]
    testSetIndexes = test_dataset.index

    print("\n\n\n These are the list of feature to train a model: ")
    for feat in list_of_impr_feat_for_H2O:
        print (feat)

    if (model_finder == "H2O"):
        [best_model_MAE, pred] = h2o_design_space_search(train_X_labels=list_of_impr_feat_for_H2O,
                                                         train_y_label=target_feat,
                                                         train_dataset=train_dataset,
                                                         test_X=test_X,
                                                         max_runtime_secs=max_run_time
                                                         )

        print("H2O best model MAE is:%s", best_model_MAE)
        if (len(pred) != len(test_y)):
            pred = pred[1:]
    elif (model_finder == "AutoML"):
        autoML_best_model = autoML_find_best_model(train_X_labels=list_of_impr_feat_for_H2O,
                                                   train_y_label=target_feat,
                                                   train_dataset=train_dataset,
                                                   test_X=test_X,
                                                   processing_time_in_secs=max_run_time
                                                   )

        pred = autoML_best_model.predict(test_X)
    elif (model_finder == 'multi_models'):
        multi_models(train_X_labels=list_of_impr_feat_for_H2O, train_y_label=target_feat, train_dataset=train_dataset,
                     test_X=test_X)

        #
        print("actual %s is: %s" % (target_feat, list(test_y.values)))

    print("test_y length is: %s" % (str(len(test_y))))
    print("pred length is: %s" % (str(len(pred))))

    [mae, model_acc_on_test] = report_error(predictions=np.array([float(x) for x in pred]),
                                            y_infer_actual_yield=np.array([float(x) for x in test_y]))

    pred = pred + np.array(bias * mae)
    pred = change_the_slope_of_graph(pred, slope)
    pred = shift(pred, num_week_shift, cval=mae)

    if (match_with_actual_yield == True):
        pred = match_with_actual(pred, test_y)

    [mae, model_acc_on_test] = report_error(predictions=np.array([float(x) for x in pred]),
                                            y_infer_actual_yield=np.array([float(x) for x in test_y]))

    [manual_mae, manual_acc] = report_error(predictions=np.array([float(x) for x in test_manual_preds]),
                                            y_infer_actual_yield=np.array([float(x) for x in test_y]))

    print("Actual Yield" + str(test_y))
    print("Pred Yield" + str(pred))

    plot(
        y_infer_actual_yield=np.array([float(x) for x in test_y]),
        inferDataSetIndex=testSetIndexes,
        modelAccuracy=model_acc_on_test,
        growerAccuracy=manual_acc,
        predictions=np.array([float(x) for x in pred]),
        y_infer_pred_yield=np.array([float(x) for x in test_manual_preds]),
        fig_name="Num_of_Most_Corr_Feat_" + str(num_of_most_corr_feats) + "Slope_" + str(
            slope) + "_max_run_time_" + str(max_run_time) + str(".eps")
    )


def multi_models(train_X_labels, train_y_label, train_dataset, test_X):
    bext = ExtraTreesRegressor(bootstrap=False, criterion='mae', max_depth=None,
                               max_features='sqrt', max_leaf_nodes=None,
                               min_impurity_split=1e-07, min_samples_leaf=1,
                               min_samples_split=2, min_weight_fraction_leaf=0.0,
                               n_estimators=200, n_jobs=1, oob_score=False, random_state=42,
                               verbose=0,
                               warm_start=False,
                               )

    ll = LassoLars(alpha=0.0001, copy_X=True, eps=2.220446049250313e-16,
                   fit_intercept=True, fit_path=True, max_iter=500, normalize=True,
                   positive=False, precompute='auto', verbose=False)

    my_cv1 = ShuffleSplit(n_splits=10, test_size=0.10, random_state=0)

    my_cv2 = TimeSeriesSplit(n_splits=10).split(train_dataset[train_X_labels])

    scoring_fnc1 = make_scorer(mean_absolute_error)

    scoring_fnc2 = make_scorer(r2_score)

    pipe8 = Pipeline([
        ('norm', Normalizer()),
        ('feature', SelectFromModel(ll)),
        ('model', ll)
    ])

    pipe9 = Pipeline([
        ('feature', SelectFromModel(ll)),
        ('model', ll)
    ])

    pipe10 = Pipeline([
        ('norm', StandardScaler()),
        ('feature', SelectFromModel(ll)),
        ('model', ll)
    ])

    pll = {"model__alpha": [0, 0.5, 1e0, 1e-2, 1e-4, 1e-6]
           }

    pipe1 = Pipeline([
        ('norm', Normalizer()),
        ('feature', SelectFromModel(bext)),
        ('model', bext)
    ])

    pipe11 = Pipeline([
        ('norm', StandardScaler()),
        ('feature', SelectFromModel(bext)),
        ('model', bext)
    ])

    pipe2 = Pipeline([
        ('feature', SelectFromModel(bext)),
        ('model', bext)
    ])

    pipe22 = Pipeline([
        ('norm', Normalizer()),
        ('model', bext)
    ])

    pipe3 = Pipeline([
        ('model', bext)
    ])

    from sklearn.feature_selection import mutual_info_regression

    pipe4 = Pipeline([
        ('norm', Normalizer()),
        ('feature', SelectKBest(score_func=mutual_info_regression, k=14)),
        ('model', ll)
    ])

    pipe5 = Pipeline([
        ('norm', Normalizer()),
        ('model', ll)
    ])

    pipe6 = Pipeline([
        ('feature', SelectKBest(score_func=mutual_info_regression, k=20)),
        ('model', ll)
    ])

    pipe7 = Pipeline([
        ('model', ll)
    ])

    pext = {'model__criterion': ['mse', 'mae'],
            'model__max_features': ['auto', 'sqrt'],
            'model__n_estimators': [40, 50, 100, 200, 300, 500]}

    scoring_fnc1 = make_scorer(mean_absolute_error)

    pipe_list_extra_tree_Reg = [pipe1, pipe11, pipe2, pipe22, pipe3]
    pipe_list_extra_lasso = [pipe7, pipe6, pipe5, pipe4, pipe10, pipe9, pipe8]
    diff_models_results = []
    for pipe in pipe_list_extra_tree_Reg:
        try:
            clf = GridSearchCV(estimator=pipe, param_grid=pext, cv=my_cv1, scoring=scoring_fnc2, n_jobs=-1)
            clf = clf.fit(train_dataset[train_X_labels].values, train_dataset[train_y_label].values)
            diff_models_results.append(
                clf.best_estimator_.fit(train_dataset[train_X_labels], train_dataset[train_y_label]).predict(test_X))
            print("pipeline model result is: %s" % (
                clf.best_estimator_.fit(train_dataset[train_X_labels], train_dataset[train_y_label]).predict(test_X)))
        except ValueError:
            print(ValueError)

    for pipe in pipe_list_extra_lasso:
        try:
            clf = GridSearchCV(estimator=pipe, param_grid=pll, cv=my_cv1, scoring=scoring_fnc2, n_jobs=-1)
            clf = clf.fit(train_dataset[train_X_labels].values, train_dataset[train_y_label].values)
            diff_models_results.append(
                clf.best_estimator_.fit(train_dataset[train_X_labels], train_dataset[train_y_label]).predict(test_X))
            print("pipeline model result is: %s" % (
                clf.best_estimator_.fit(train_dataset[train_X_labels], train_dataset[train_y_label]).predict(test_X)))
        except ValueError:
            print(ValueError)

    print("the result for diff models are:")
    for list in diff_models_results:
        print(list)


def match_with_actual(pred, actual_yield):
    matched_pred = []
    matched_pred.append(actual_yield.iloc[0])
    for i in range(1, len(actual_yield)):
        matched_pred.append(pred[i] - pred[i - 1] + actual_yield.iloc[i - 1])

    return matched_pred


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

    parser.add_argument("--train_set_size", action="store", dest="train_set_size", type=float,
                        help="This argument determines the percentage of dataset that is dedicated to train_set (e.g., 0.9 means the first 90 precent chunk of dataset is dedicated to training set).")
    parser.add_argument("--dataset_file_addr", action="store", dest="dataset_file_add", type=str,
                        help="This should point to a climate/yield file that you can find an example file in the current folder. note that the sheet-name has to be entire_data.")
    parser.add_argument("--oprs_for_feat_eng", action="store", dest="oprs_for_feat_eng", type=str,
                        help="This argument determines the operations that should be used in feat engineering (e.g., \"[\"Add\", \"Sub\", \"Exp\", \"Sin\", \"Cos\", \"Mult\", \"Div\"]\", or \"mult\").\n Be careful about quotations when you make the list of feat_engin operations.")
    parser.add_argument("--max_run_time", action="store", dest="max_run_time", type=int,
                        help="This argument determines the time that we want to spend on model finding/tuning (e.g., 3600, means one hour).")
    parser.add_argument("--num_of_most_corr_feats", action="store", dest="num_of_most_corr_feats", type=int,
                        help="This argument determines the number of most important features (parameters) that are selected from feature engineered dataset (e.g., 20).")
    parser.add_argument("--feature_select_methods", action="store", dest="feature_select_methods", type=str,
                        help="This argument determines the feature selection method (it could be ""feat-import"", ""rec-feat-elimin"", ""univar-select"", or ""most-corr"").")
    parser.add_argument("--verbose", '-v', action="store", dest="verbose", type=int,
                        help="This argument determines the verbosity level (it could be 0, 5, 10).")
    parser.add_argument("--model_finder", action="store", dest="model_finder", type=str,
                        help="This argument determines which automated model finder algorithm should be deployed, automl, or h20.")
    parser.add_argument("--dataset_stat_visual_flag", type=str2bool, nargs='?', dest="dataset_stat_visual_flag")
    parser.add_argument("--rmv_feat_with_corr_higher_than", type=float, nargs='?', dest="rmv_feat_with_corr_higher_than", help="We determine the correlation between every two features before feature engineering and if they are higher than ""rmv_feat_with_corr_higher_than"", they will be removed (e.g., rmv_feat_with_corr_higher_than = 1 means do not remove any feature).")
    parser.add_argument("--slope", type=float, nargs='?',
                        dest="slope",
                        help="This determines how to chnage the slope between every two consecuitive pred-points (e.g., 1.8 means to increase the slope, while -1.8 means decrease the slope, and one means does not change the slope). ")
    parser.add_argument("--bias", type=float, nargs='?',
                        dest="bias",
                        help="This determines how much bias give to all the pred points (pred = pred + bias*mae)")
    parser.add_argument("--target_feat", type=str, nargs='?',
                        dest="target_feat",
                        help="This determines what the label of a column is that correspond to target feat.")

    parser.add_argument("--drop_feat", type=str, nargs='?',
                        dest="drop_feat",
                        help="This determines which feature should be dropped before any process.")

    parser.add_argument("--num_week_shift", type=int, nargs='?',
                        dest="num_week_shift",
                        help="This determines the predictions the prediction should be shiftted how many weeks.")

    parser.add_argument("--alr_gen_feats", type=str2bool, nargs='?', dest="alr_gen_feats",
                        help="This determines if we already did feature engineering and sorted the most important features in order not to repeat this process.")

    parser.add_argument("--add_mov_win_rolling_feats", type=str2bool, nargs='?', dest="add_mov_win_rolling_feats",
                        help="This argument determines if we want to add moving window rolling features to the dataset or not.")

    parser.add_argument("--match_with_actual_yield", type=str2bool, nargs='?', dest="match_with_actual_yield",
                        help="This argument determines if we want match the actual yield with predicted yield to perform next week predicion.")

    parser.add_argument("--intel_feat_moving", type=str2bool, nargs='?', dest="intel_feat_moving",
                        help="This argument determines if we want to activate intelligent feat moving for the features or not.")

    parser.add_argument("--num_procs", type=int, nargs='?', dest="num_procs",
                        help="This argument limit the number of processes that this program runs in parallel.")


    # read arguments from the command line
    args = parser.parse_args()

    main()
