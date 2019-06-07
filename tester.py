"------------------------------------"
"------------Tester Code-------------"
"--1st Stage: No Sky consideration---"
"------------------------------------"

import numpy as np
import data_read as reader
import ml_dec_tree as DT
import ml_svm as SVM
import xlsxwriter
import time
import copy


"Get average errors as labels for the tree"
def getAvgLabel(nodes_errors):

    nodes_label_avg = list()

    for i in nodes_errors:

        node = i['node']
        list_errors_i = i['errors']
        avg_i = np.average(list_errors_i)

        sample = {
            'node':node,
            'label_avg':avg_i
            }

        nodes_label_avg.append(sample)

    return nodes_label_avg


"Calculate the weighted average error of the whole tree in tree"
def rmse_total(label_avg,nodes_errors):

    nodes_mse = 0
    nodes_n = 0

    for i in nodes_errors:

        for j in label_avg:

            node_avg = j['node']
            node_error = i['node']

            if node_avg == node_error:

                avg_val = j['label_avg']
                list_errors_i = i['errors']
                list_errors_i_n = len(list_errors_i)
                list_diff_i = [x - avg_val for x in list_errors_i]
                list_squared_diff_i = [x**2 for x in list_diff_i]
                sum_squares = sum(list_squared_diff_i)
                nodes_n = nodes_n + list_errors_i_n
                nodes_mse = nodes_mse + sum_squares

    avg_mse = nodes_mse/nodes_n
    avg_rmse = np.sqrt(avg_mse)

    return avg_rmse


"Calculate the weighted average error of the whole tree for SVR algorithm"
def rmse_svr(target_errors,prediction):
    rmse = np.sqrt(((prediction - target_errors) ** 2).mean())
    return rmse


"Calculate the weighted average error of the whole tree in tree"
def errorsInMargin2(nodes_labels,nodes_errors):

    count_correct = 0
    count_total = 0
    correct_percentage = 0
    nodes_errors_in_margin = list()
    nodes_pred_val_true_val = list()

    for i in nodes_labels:

        node_label = i['node']

        for j in nodes_errors:

            node_error = j['node']

            if node_label == node_error:

                min_val_label = i['label']['min']
                max_val_label = i['label']['max']
                list_errors = j['errors']
                count_total_node = len(list_errors)
                count_correct_node = sum(min_val_label < x < max_val_label for x in list_errors)

                sample = {
                    'node':node_label,
                    'label':i['label'],
                    'correct':count_correct_node,
                    'total':count_total_node,
                    'correct_%':count_correct_node/count_total_node
                    }

                nodes_errors_in_margin.append(sample)

                node_error = j['errors']

                for k in node_error:

                    sample = {
                        'node':node_label,
                        'pred_val':max_val_label,
                        'true_val':k
                        }

                    nodes_pred_val_true_val.append(sample)

                count_correct = count_correct + count_correct_node
                count_total = count_total + count_total_node

    correct_percentage = count_correct/count_total

    return nodes_errors_in_margin,nodes_pred_val_true_val,correct_percentage


"Joins all the label nodes from the trained tree to make one single listing of all labels required"
def mergeTrainTestLabels(train_labels,train_errors,test_labels,test_errors):

    train_labels_copy = copy.deepcopy(train_labels)

    label_avg_train = getAvgLabel(train_errors)

    for i in test_labels:

        if i not in train_labels:

            train_labels_copy.append(i)
            node_test_missing = i['node']

            for j in test_labels:

                if j['node'] == node_test_missing:

                    node_test_missing_max = j['label']['max']
                    node_test_missing_min = j['label']['min']
                    avg_missing = (node_test_missing_max+node_test_missing_min)/2

                    sample = {
                    'node':node_test_missing,
                    'label_avg':avg_missing
                    }

                    label_avg_train.append(sample)

    return train_labels_copy,label_avg_train


"Decision tree method to call required information and generate the tree"
def decisionTree(filter_input_train,filter_output_train,filter_errors_train,input_test,output_test):

    possible_bins = [100] #[10,20,50,100,200]
    possible_segmentation = [6] #[2,4,6]
    possible_tree_depth = [10] #[4,5,6,7,8,9,10]
    results_train = list()
    results_test = list()

    "Calculate statistics and labels for all features"
    features = {
        'cno':DT.calculateStatisticsSatellites(filter_input_train,'cno'),
        'lsq_residual':DT.calculateStatisticsSatellites(filter_input_train,'lsq_residual'),
        'elevation':DT.calculateStatisticsSatellites(filter_input_train,'elevation'),
        'azimuth':DT.calculateStatisticsSatellites(filter_input_train,'azimuth'),
        'pdop':DT.calculateStatistics(filter_input_train,'pdop'),
        'ndop':DT.calculateStatistics(filter_input_train,'ndop'),
        'nr_used_measurements':DT.indicateLabel(filter_input_train,'nr_used_measurements'),
        'constellation':DT.indicateLabelSatellites(filter_input_train,'constellation'),
        'tracking_type':DT.indicateLabelSatellites(filter_input_train,'tracking_type'),
        'multipath':DT.indicateLabelSatellites(filter_input_train,'multipath'),
        'raim':DT.indicateLabelSatellites(filter_input_train,'raim'),
        'cycle_slip':DT.indicateLabelSatellites(filter_input_train,'cycle_slip'),
        'difference_ENU_e':DT.calculateStatisticsEkf(filter_input_train,'difference_ENU_e'),
        'difference_ENU_n':DT.calculateStatisticsEkf(filter_input_train,'difference_ENU_n'),
        'difference_ENU_u':DT.calculateStatisticsEkf(filter_input_train,'difference_ENU_u'),
        'innovation_ENU_e':DT.calculateStatisticsEkf(filter_input_train,'innovation_ENU_e'),
        'innovation_ENU_n':DT.calculateStatisticsEkf(filter_input_train,'innovation_ENU_n'),
        'innovation_ENU_u':DT.calculateStatisticsEkf(filter_input_train,'innovation_ENU_u')
        }

    for i in possible_bins:

        for j in possible_segmentation:

            for k in possible_tree_depth:

                "Training of the Decision Tree"
                classification_error_train,tree_classifier_train,nodes_labels_train,nodes_errors_train,input_test,output_test,input_train,output_train,divisions,average_per_bin,features,tree_train = DT.tree(filter_input_train,filter_output_train,filter_errors_train,input_test,output_test,features,i,j,fraction,k)

                "Training of the Decision Tree"
                classification_error_test,nodes_labels_test,nodes_errors_test,tree_test = DT.testTree(input_test,output_test,i,j,tree_classifier_train,nodes_labels_train,divisions,average_per_bin,features,k)

                nodes_labels_total,label_avg_total = mergeTrainTestLabels(nodes_labels_train,nodes_errors_train,nodes_labels_test,nodes_errors_test)

                train_rmse =  rmse_total(label_avg_total,nodes_errors_train)
                test_rmse = rmse_total(label_avg_total,nodes_errors_test)
                nodes_errors_in_margin_train,nodes_pred_val_train,correct_percentage_train = errorsInMargin2(nodes_labels_total,nodes_errors_train)
                nodes_errors_in_margin,nodes_pred_val,correct_percentage = errorsInMargin2(nodes_labels_total,nodes_errors_test)

                sample = {
                    'bins':i,
                    'segmentation':j,
                    'tree_depth':k,
                    'classification error':classification_error_train,
                    'average rmse':train_rmse,
                    'correct_percentage':correct_percentage_train,
                    'tree':tree_classifier_train
                    }

                results_train.append(sample)

                sample = {
                    'bins':i,
                    'segmentation':j,
                    'tree_depth':k,
                    'classification error':classification_error_test,
                    'average rmse':test_rmse,
                    'correct_percentage':correct_percentage,
                    'nodes_pred_val':nodes_pred_val,
                    'tree':tree_classifier_train
                    }

                results_test.append(sample)

                print("bin: "+str(i)+" "+"segment: "+str(j)+" "+"depth: "+str(k))


    "From here: Print out of results"

    "Generates a workbook to print the data to"
    workbook = xlsxwriter.Workbook('name.xlsx')

    worksheet = workbook.add_worksheet("train")

    for i in range(0,len(results_train)):

        worksheet.write(i,0, results_train[i]['bins'])
        worksheet.write(i,1, results_train[i]['segmentation'])
        worksheet.write(i,2, results_train[i]['tree_depth'])
        worksheet.write(i,3, results_train[i]['classification error'])
        worksheet.write(i,4, results_train[i]['average rmse'])
        worksheet.write(i,5, results_train[i]['correct_percentage'])

    for i in range(0,len(results_train)):

        worksheet = workbook.add_worksheet("tree "+"b "+str(results_train[i]['bins'])+" "+"s "+str(results_train[i]['segmentation'])+" "+"td "+str(results_train[i]['tree_depth']))

        for j in range(0,len(results_train[i]['tree'])):

            node = results_train[i]['tree'][j]['node']
            feature = results_train[i]['tree'][j]['feature']
            worksheet.write(j,0, node)
            worksheet.write(j,1, feature)

    worksheet = workbook.add_worksheet("Features")
    possible_segmentation = possible_segmentation[0]
    worksheet.write(0,0,'cno')
    avg_cno = features['cno'][0]
    std_dev_cno = features['cno'][1]
    worksheet.write(0,1,avg_cno)
    worksheet.write(0,2,std_dev_cno)
    worksheet.write(0,3,str(DT.createGroupsContinuous(possible_segmentation,avg_cno,std_dev_cno)))

    worksheet.write(1,0,'lsq_residual')
    avg_lsq_residual = features['lsq_residual'][0]
    std_dev_lsq_residual = features['lsq_residual'][1]
    worksheet.write(1,1,avg_lsq_residual)
    worksheet.write(1,2,std_dev_lsq_residual)
    worksheet.write(1,3,str(DT.createGroupsContinuous(possible_segmentation,avg_lsq_residual,std_dev_lsq_residual)))

    worksheet.write(2,0,'elevation')
    avg_elevation = features['elevation'][0]
    std_dev_elevation = features['elevation'][1]
    worksheet.write(2,1,avg_elevation)
    worksheet.write(2,2,std_dev_elevation)
    worksheet.write(2,3,str(DT.createGroupsContinuous(possible_segmentation,avg_elevation,std_dev_elevation)))

    worksheet.write(3,0,'azimuth')
    avg_azimuth = features['azimuth'][0]
    std_dev_azimuth = features['azimuth'][1]
    worksheet.write(3,1,avg_azimuth)
    worksheet.write(3,2,std_dev_azimuth)
    worksheet.write(3,3,str(DT.createGroupsContinuous(possible_segmentation,avg_azimuth,std_dev_azimuth)))

    worksheet.write(4,0,'pdop')
    avg_pdop = features['pdop'][0]
    std_dev_pdop = features['pdop'][1]
    worksheet.write(4,1,avg_pdop)
    worksheet.write(4,2,std_dev_pdop)
    worksheet.write(4,3,str(DT.createGroupsContinuous(possible_segmentation,avg_pdop,std_dev_pdop)))

    worksheet.write(5,0,'ndop')
    avg_ndop = features['ndop'][0]
    std_dev_ndop = features['ndop'][1]
    worksheet.write(5,1,avg_ndop)
    worksheet.write(5,2,std_dev_ndop)
    worksheet.write(5,3,str(DT.createGroupsContinuous(possible_segmentation,avg_ndop,std_dev_ndop)))

    worksheet.write(6,0,'nr_used_measurements')
    worksheet.write(6,1,str(features['nr_used_measurements']))

    worksheet.write(7,0,'constellation')
    worksheet.write(7,1,str(features['constellation']))

    worksheet.write(8,0,'tracking_type')
    worksheet.write(8,1,str(features['tracking_type']))

    worksheet.write(9,0,'multipath')
    worksheet.write(9,1,str(features['multipath']))

    worksheet.write(10,0,'raim')
    worksheet.write(10,1,str(features['raim']))

    worksheet.write(10,0,'cycle_slip')
    worksheet.write(10,1,str(features['cycle_slip']) )

    worksheet.write(11,0,'difference_ENU_e')
    avg_difference_ENU_e = features['difference_ENU_e'][0]
    std_dev_difference_ENU_e = features['difference_ENU_e'][1]
    worksheet.write(11,1,avg_difference_ENU_e)
    worksheet.write(11,2,std_dev_difference_ENU_e)
    worksheet.write(11,3,str(DT.createGroupsContinuous(possible_segmentation,avg_difference_ENU_e,std_dev_difference_ENU_e)))

    worksheet.write(12,0,'difference_ENU_n')
    avg_difference_ENU_n = features['difference_ENU_n'][0]
    std_dev_difference_ENU_n = features['difference_ENU_n'][1]
    worksheet.write(12,1,avg_difference_ENU_n)
    worksheet.write(12,2,std_dev_difference_ENU_n)
    worksheet.write(12,3,str(DT.createGroupsContinuous(possible_segmentation,avg_difference_ENU_n,std_dev_difference_ENU_n)))

    worksheet.write(13,0,'difference_ENU_u')
    avg_difference_ENU_u = features['difference_ENU_u'][0]
    std_dev_difference_ENU_u = features['difference_ENU_u'][1]
    worksheet.write(13,1,avg_difference_ENU_u)
    worksheet.write(13,2,std_dev_difference_ENU_u)
    worksheet.write(13,3,str(DT.createGroupsContinuous(possible_segmentation,avg_difference_ENU_u,std_dev_difference_ENU_u)))

    worksheet.write(14,0,'innovation_ENU_e')
    avg_innovation_ENU_e = features['innovation_ENU_e'][0]
    std_dev_innovation_ENU_e = features['innovation_ENU_e'][1]
    worksheet.write(14,1,avg_innovation_ENU_e)
    worksheet.write(14,2,std_dev_innovation_ENU_e)
    worksheet.write(14,3,str(DT.createGroupsContinuous(possible_segmentation,avg_innovation_ENU_e,std_dev_innovation_ENU_e)))

    worksheet.write(15,0,'innovation_ENU_n')
    avg_innovation_ENU_n = features['innovation_ENU_n'][0]
    std_dev_innovation_ENU_n = features['innovation_ENU_n'][1]
    worksheet.write(15,1,avg_innovation_ENU_n)
    worksheet.write(15,2,std_dev_innovation_ENU_n)
    worksheet.write(15,3,str(DT.createGroupsContinuous(possible_segmentation,avg_innovation_ENU_n,std_dev_innovation_ENU_n)))

    worksheet.write(16,0,'innovation_ENU_u')
    avg_innovation_ENU_u = features['innovation_ENU_u'][0]
    std_dev_innovation_ENU_u = features['innovation_ENU_u'][1]
    worksheet.write(16,1,avg_innovation_ENU_u)
    worksheet.write(16,2,std_dev_innovation_ENU_u)
    worksheet.write(16,3,str(DT.createGroupsContinuous(possible_segmentation,avg_innovation_ENU_u,std_dev_innovation_ENU_u)))

    worksheet = workbook.add_worksheet("all_tests")

    for i in range(0,len(results_test)):

        worksheet.write(i,0, results_test[i]['bins'])
        worksheet.write(i,1, results_test[i]['segmentation'])
        worksheet.write(i,2, results_test[i]['tree_depth'])
        worksheet.write(i,3, results_test[i]['classification error'])
        worksheet.write(i,4, results_test[i]['average rmse'])
        worksheet.write(i,5, results_test[i]['correct_percentage'])

    for i in range(0,len(results_test)):

        worksheet = workbook.add_worksheet("test "+"bin "+str(results_test[i]['bins'])+" segment "+str(results_test[i]['segmentation'])+" td "+str(results_test[i]['tree_depth']))
        worksheet.write(i,0, results_test[i]['bins'])
        worksheet.write(i,1, results_test[i]['segmentation'])
        worksheet.write(i,2, results_test[i]['tree_depth'])
        worksheet.write(i,3, results_test[i]['classification error'])
        worksheet.write(i,4, results_test[i]['average rmse'])
        worksheet.write(i,5, results_test[i]['correct_percentage'])

        for j in range(0,len(results_test[i]['nodes_pred_val'])):

            worksheet.write(j,6, results_test[i]['nodes_pred_val'][j]['node'])
            worksheet.write(j,7, results_test[i]['nodes_pred_val'][j]['pred_val'])
            worksheet.write(j,8, results_test[i]['nodes_pred_val'][j]['true_val'])

    workbook.close()


"Support Vector Machines method to call required information, and generate the trained SVR model"
def svr(filter_input_train_svr,filter_output_train_svr,filter_errors_train_svr,input_test_svr,output_test_svr):

    "Data reading function specifically designed for Support Vector Regression capability"
    "SVM requires all input data features to be numerical in order to calculate kernel and proceed mathematically"
    "Changes Constellations and Tracking String labels into numerical values"
    filter_input_train_svr = SVM.svrConstellationTrackingChange(filter_input_train_svr)
    input_test_svr = SVM.svrConstellationTrackingChange(input_test_svr)

    "Features for inclusion in the SVM algorithm. Change these features and set them in the order of relevance according to information Gain"
    all_features_include = ['difference_ENU','elevation','nr_used_measurements','cno','lsq_residual','azimuth','innovation_ENU','constellation','pdop','ndop']

    "Kernel selection"
    kernel = ['rbf'] #['rbf','poly']

    "epsilon: Deviation from targets allowed"
    epsilon_i = [0.001] #[0.0001,0.001,0.01]

    "gamma: defines area of influence of single training inputs. Higher gamma, less area of influence. if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma"
    gamma_i = [5] #[0.5,1,2,5]

    "Definition of C: Positive constant to determine slack of alpha"
    C_i = [1] #[1,2,5,10]

    "Opens Workbook:"
    workbook = xlsxwriter.Workbook('Stage_2_corrected_SVM_results_all_data_average_rank_no_sky_5.xlsx')

    "Creates result vector to print results later"
    results = list()

    for f in [7]:


        print("started batch of features")

        feat = all_features_include[:f+1]

        filter_input_train_svr_rm = SVM.svrRemoveFeature(copy.deepcopy(filter_input_train_svr),feat)
        input_test_svr_rm = SVM.svrRemoveFeature(copy.deepcopy(input_test_svr),feat)

        for k in kernel:

            if k == 'rbf':

                "Degree of Polynomial kernel (if used)"
                degree = [1]

            else:

                "Degree of Polynomial kernel (if used)"
                degree = [2,3,4]

            for e in epsilon_i:

                for g in gamma_i:

                    for c in C_i:

                        for d in degree:

                            start_svr_iter = time.time()

                            target_errors,prediction = SVM.mainSVR(filter_input_train_svr_rm,filter_output_train_svr,filter_errors_train_svr,input_test_svr_rm,output_test_svr,k,d,e,g,c,feat)

                            end_svr_iter = time.time()
                            diff_svr_iter = end_svr_iter - start_svr_iter
                            print("Finished SVM main k "+str(k)+" "+"e "+str(e)+" "+"g "+str(g)+" "+"c "+str(c)+" "+"d "+str(d)+" (s): "+str(diff_svr_iter))
                            print("Finished SVM main k "+str(k)+" "+"e "+str(e)+" "+"g "+str(g)+" "+"c "+str(c)+" "+"d "+str(d)+" (min): "+str(diff_svr_iter/60))

                            start_svr_save_iter = time.time()

                            rmse_svr_iter = rmse_svr(target_errors,prediction)

                            sample = {
                                'slack':c,
                                'gamma':g,
                                'epsilon':e,
                                'features':f,
                                'kernel':k,
                                'average rmse':rmse_svr_iter
                                }

                            results.append(sample)

                            worksheet = workbook.add_worksheet("f "+str(f)+" "+"k "+str(k)+" "+"e "+str(e)+" "+"g "+str(g)+" "+"c "+str(c))
                            worksheet.write(0,0, "Features: "+" "+str(feat))

                            for i in range(0,len(prediction)):

                                worksheet.write(i+1,0, prediction[i])
                                worksheet.write(i+1,1, target_errors[i])

                            worksheet.write(0,1,"rmse: "+str(rmse_svr_iter))

                            end_svr_save_iter = time.time()
                            diff_svr_save_iter = end_svr_save_iter - start_svr_save_iter
                            print("Finished calculating error and printing results (s): "+str(diff_svr_save_iter))
                            print("Finished calculating error and printing results (min): "+str(diff_svr_save_iter/60))

                            print("finished iteration: f "+str(f)+"K "+str(k)+" "+"e "+str(e)+" "+"g "+str(g)+" "+"c "+str(c)+" "+"d "+str(d))

        print("finished batch of features")

    worksheet = workbook.add_worksheet("rmse_results")
    worksheet.write(0,0, 'kernel')
    worksheet.write(0,1, 'epsilon')
    worksheet.write(0,2, 'gamma')
    worksheet.write(0,3, 'slack')
    worksheet.write(0,4, 'average rmse')
    worksheet.write(0,5, 'features')

    for i in range(0,len(results)):

        worksheet.write(i+1,0, results[i]['kernel'])
        worksheet.write(i+1,1, results[i]['epsilon'])
        worksheet.write(i+1,2, results[i]['gamma'])
        worksheet.write(i+1,3, results[i]['slack'])
        worksheet.write(i+1,4, results[i]['average rmse'])
        worksheet.write(i+1,5, results[i]['features'])

    workbook.close()


"Defines the information to be used for algorithms to train and test on:"
possible_input = ['inputFeaturesFile1','inputFeaturesFile2']
possible_output = ['outputFeaturesFile1','outputFeaturesFile2']

"Defines the fraction of data to be used for testing. Uses 20% of the data for testing"
fraction = 0.2

"time.time() is called to assess time and code execution. It is called several times"
start_read_time = time.time()

"Obtain the vector of input and output data with already filtered values"
filter_input_train,filter_output_train,filter_errors_train,input_test,output_test = reader.read(possible_input,possible_output,fraction)

end_read_time = time.time()
diff_read_time = end_read_time - start_read_time
print ("reading time (s): "+str(diff_read_time))
print ("reading time (min): "+str(diff_read_time/60))

"If 1 is in selector, runs Decision Tree, if 2 is in selector runs SVR. 1 and 2 may be written inside the brackets to run both algorithms"
selector = [2]

if 1 in selector:

    start_train_test_time = time.time()
    decisionTree(filter_input_train,filter_output_train,filter_errors_train,input_test,output_test)
    end_train_test_time = time.time()
    diff_train_test_time = end_train_test_time - start_train_test_time
    print ("Decision Tree training and testing time (s): "+str(diff_train_test_time))
    print ("Decision Tree training and testing time (min): "+str(diff_train_test_time/60))

if 2 in selector:

    start_train_test_time = time.time()

    filter_input_train_svr = copy.deepcopy(filter_input_train)
    filter_output_train_svr = copy.deepcopy(filter_output_train)
    input_test_svr = copy.deepcopy(input_test)
    output_test_svr = copy.deepcopy(output_test)
    filter_errors_train_svr = copy.deepcopy(filter_errors_train)

    svr(filter_input_train_svr,filter_output_train_svr,filter_errors_train_svr,input_test_svr,output_test_svr)
    end_train_test_time = time.time()
    diff_train_test_time = end_train_test_time - start_train_test_time
    print ("Support Vector Machine training and testing time (s): "+str(diff_train_test_time))
    print ("Support Vector Machine training and testing time (min): "+str(diff_train_test_time/60))
