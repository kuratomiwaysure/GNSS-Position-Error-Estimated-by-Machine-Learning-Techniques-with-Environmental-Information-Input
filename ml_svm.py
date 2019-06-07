"-------------------------------------------"
"Support Vector Machines Algorithm main file"
"------1st Stage: No Sky consideration------"
"-------------------------------------------"

import numpy as np
from scipy . optimize import minimize
import data_read as reader
from sklearn.svm import SVR


"General calculate statistics method for general"
def calculateStatistics(dataset,string):

    result = list()
    [result.append(x[string]) for x in dataset]
    average = np.average(result)
    std_dev = np.std(result)

    return average,std_dev


"Counts the number of satellites in single time input corresponding to every feature. Whatever is the most important feature, will determine the label"
def mostCommonPerFeature(dataset,feature):

    feature_hits = list()
    [feature_hits.append(i[feature]) for i in dataset]
    most_common = max(set(feature_hits), key=feature_hits.count)

    return most_common


"Data reading function specifically designed for Support Vector Regression capability"
"SVR requires all input data features to be numerical in order to calculate kernel and proceed mathematically"
"Changes Constellations and Tracking String labels into numerical values"
def svrConstellationTrackingChange(input_data):

    for i in range(0,len(input_data)):

        satellites_list = input_data[i]['used_satellites']

        for j in range(0,len(satellites_list)):

            constellation = satellites_list[j]['constellation']
            tracking = satellites_list[j]['tracking_type']

            if constellation == 'gps':

                input_data[i]['used_satellites'][j]['constellation'] = 1

            if constellation == 'glonass':

                input_data[i]['used_satellites'][j]['constellation'] = 2

            if constellation == 'galileo':

                input_data[i]['used_satellites'][j]['constellation'] = 3

            if constellation == 'beidou':

                input_data[i]['used_satellites'][j]['constellation'] = 4

            if tracking == 'pll':

                input_data[i]['used_satellites'][j]['tracking_type'] = 1

            if tracking == 'costas':

                input_data[i]['used_satellites'][j]['tracking_type'] = 2

            if tracking == 'no_tracking':

                input_data[i]['used_satellites'][j]['tracking_type'] = 3

    return input_data


"Remove all features from the input vectors that are not of interest"
def svrRemoveFeature(input_data,features_interest):

    all_features = ['pdop','ndop','nr_used_measurements','correction_covariance','correction_state','correction_state_ENU','difference_ENU','innovation_ENU','innovation','prediction_covariance','prediction_state','prediction_state_ENU','cno','constellation','tracking_type','elevation','azimuth','lsq_residual','multipath','raim','cycle_slip']

    for i in features_interest:

        all_features.remove(i)

    for i in input_data:

        sat_i = i['used_satellites']
        ekf_i = i['ekf_info']

        for j in all_features:

            if j in i:

                del i[j]

            if j in ekf_i:

                del ekf_i[j]

        for k in sat_i:

            for m in all_features:

                if m in k:

                    del k[m]

    return input_data


"Changes the input and output data into proper vectors (arrays) for multiplication"
def svrDataVectors(input_data,target,all_features_interest):

    x = list()
    y = target
    ind = input_data
    sample_in = list()

    for i in range(0,len(ind)):

        ekf_i = ind[i]['ekf_info']

        for j in all_features_interest:

            if j == 'pdop' or j == 'ndop' or j == 'nr_used_measurements':

                feature = ind[i][j]
                sample_in.append(feature)

            if j == 'difference_ENU':

                sample_in.append(ekf_i['difference_ENU']['e'])
                sample_in.append(ekf_i['difference_ENU']['n'])
                sample_in.append(ekf_i['difference_ENU']['u'])

            if j == 'innovation_ENU':

                sample_in.append(ekf_i['innovation_ENU']['e'])
                sample_in.append(ekf_i['innovation_ENU']['n'])
                sample_in.append(ekf_i['innovation_ENU']['u'])

            if j == 'cno' or j == 'elevation' or j == 'azimuth' or j == 'lsq_residual':

                avg,std = calculateStatistics(input_data[i]['used_satellites'],j)
                sample_in.append(avg)

            elif j == 'constellation' or j == 'tracking_type' or j == 'multipath' or j == 'raim' or j == 'cycle_slip':

                common = mostCommonPerFeature(input_data[i]['used_satellites'],j)
                sample_in.append(common)

        x.append(sample_in[:])
        sample_in = []

    x = np.asarray(x)
    y = np.asarray(y)

    return x,y


"Obtains main variables as the length of the vectors, the input vector, target vector, and the P matrix"
def svrPreparation(input_data,target,input_data_test,target_test,all_features_include):

    N = len(input_data)
    x,target = svrDataVectors(input_data,target,all_features_include)
    x_test,targets_test = svrDataVectors(input_data_test,target_test,all_features_include)

    return N,x,target,x_test,targets_test


def mainSVR(filter_input_train,filter_output_train,filter_errors_train,input_test,output_test,ker,deg,epsilon_i,gamma_i,C_i,all_features_include):

    "Get length, x (inputs), targets, and P matrix"
    length,x,targets,x_test,target_test = svrPreparation(filter_input_train,filter_errors_train,input_test,output_test,all_features_include)

    "Get Error vector only"
    target_errors = list()
    [target_errors.append(i['error']) for i in target_test]
    target_errors = np.asarray(target_errors)

    "class sklearn.svm.SVR(kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)"
    "gamma : float, optional (default=’auto’) Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’"
    "SVR implementation"
    model = SVR(kernel = ker,degree = deg,gamma = gamma_i, C = C_i, epsilon = epsilon_i)
    model.fit(x,targets)
    prediction = model.predict(x_test)

    return target_errors,prediction
