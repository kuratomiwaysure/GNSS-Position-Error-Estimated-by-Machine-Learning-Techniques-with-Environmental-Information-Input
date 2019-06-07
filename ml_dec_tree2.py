"----------------------------------"
"Decision Trees Algorithm main file"
"---2nd Stage: Sky consideration---"
"----------------------------------"

import math
import data_read as reader
import numpy as np
import time


"Auxiliary method to find how many values are inside a range"
def verifyInRange(mini,maxi,errors):
    return len(list(i for i in range(0,len(errors)) if errors[i] <= maxi and errors[i] >= mini))


"Method to create bins out of errors to create equally sized bins"
def createBins(errors,divisor_bins):

    bins = list()
    min_error = min(errors)
    max_error = max(errors)
    increment = 0.001
    ran = np.arange(min_error,max_error,increment)
    bins.append(min_error)
    size_errors = len(errors)
    number_per_bin = math.ceil(size_errors/divisor_bins)
    bin_counter = 0

    for i in range(0,len(ran)):

       count = verifyInRange(bins[bin_counter],ran[i],errors)

       if count >= number_per_bin:

           bins.append(ran[i])
           bin_counter = bin_counter + 1

    bins.append(max_error)

    return bins


"Obtain a list per each of the bins"
def listPerBin(bins,errors):

    size = 0
    list_per_bin = list()
    temporal_list = list()

    for i in range(0,len(bins)-1):

        lower_bin = bins[i]
        upper_bin = bins[i+1]

        [temporal_list.append(x) for x in errors if x >= lower_bin and x <= upper_bin]

        list_per_bin.append(temporal_list[:])
        size = size + len(temporal_list)
        temporal_list = []

    return list_per_bin


"Calculate average for each of the bins created for the tree"
def averagePerBin(list_per_bin):

    average_bins = list()

    for i in range(0,len(list_per_bin)):

        avg_per_bin = np.average(list_per_bin[i])
        std_per_bin = np.std(list_per_bin[i])
        min_val = avg_per_bin-4.5*std_per_bin
        max_val = avg_per_bin+4.5*std_per_bin

        sample = {
            'min':min_val,
            'max':max_val
                }

        average_bins.append(sample)


    return average_bins


"Calculate the entropy of a dataset based on the errors classification and attribute selection"
def calculateEntropy(errors_current,histo_original,divisions):

    n_current,edges = np.histogram(errors_current,bins=divisions,density=False)
    proba = list()
    log2Proba = list()
    n_total = sum(n_current)

    if n_total == 0:

        entropy = 0

    else:

        for x in range(0,len(histo_original)):

            prob = n_current[x]/n_total

            if prob == 1:

                key = {'location':x}
                return key

            elif prob == 0:

                proba.append(prob)
                log2Proba.append(prob)

            else:

                log2Proba.append(prob*math.log2(prob))

        entropy = -sum(log2Proba)

    return entropy


"Calculate average gain with respect to a given feature input of the dataset"
def gain(dataset,feature_string,tree,segmentation,filter_errors,histo_original,divisions,features):

    weighted = 0.0
    dataset_errors = getErrors(dataset,tree,filter_errors)
    subset_errors = list()
    initial_entropy = calculateEntropy(dataset_errors,histo_original,divisions)

    if feature_string == 'cno' or feature_string == 'lsq_residual' or feature_string == 'elevation' or feature_string == 'azimuth' or feature_string == 'pdop' or feature_string == 'ndop' or feature_string == 'sky':

        if segmentation == 2:

            feature_set = ['low','high']

        elif segmentation == 4:

            feature_set = ['very_low','low','high','very_high']

        elif segmentation == 6:

            feature_set = ['ultra_low','very_low','low','high','very_high','ultra_high']

    else:

        feature_set = features[feature_string]

    subsets = selectByFeature(dataset,feature_string,segmentation,features)

    for v in range(0,len(feature_set)):

        [subset_errors.append(filter_errors[i]) for i in subsets[v]]
        entropy_subset = calculateEntropy(subset_errors,histo_original,divisions)

        if isinstance(entropy_subset, dict):

            entropy_subset = 0

        weighted += entropy_subset * len(subset_errors)
        subset_errors = []

        if len(dataset) == 0:

            result = initial_entropy

        else:

            result = initial_entropy - weighted/len(dataset)

    return result


"Get errors of a dataset given as parameter"
def getErrors(dataset_interest,dataset,filter_errors):
    errors = list()
    [errors.append(i['error']) for i in dataset_interest]

    return errors


"Method that selects the data samples going to each node based on the feature set as input"
def selectByFeature(input_dataset,feature_string,segmentation,features):

    sub_string = feature_string[0:-2]
    feature_group = feature_string+'_group'
    subsets_temp = list()
    subsets_index = list()

    if feature_string == 'cno' or feature_string == 'lsq_residual' or feature_string == 'elevation' or feature_string == 'azimuth' or feature_string == 'pdop' or feature_string == 'ndop' or feature_string == 'sky' or sub_string == 'prediction_state' or sub_string == 'prediction_covariance' or sub_string == 'correction_state' or sub_string == 'correction_covariance' or sub_string =='difference_ENU' or sub_string == 'innovation_ENU':

        for j in range(0,segmentation):

            for i in input_dataset:

                global_index = i['global']
                group = i[feature_group]

                if j == group:

                    subsets_temp.append(global_index)

            subsets_index.append(subsets_temp[:])
            subsets_temp = []

    else:

        len_feature = len(features[feature_string])

        for j in range(0,len_feature):

            for i in input_dataset:

                global_index = i['global']
                group = i[feature_group]

                if j == group:

                    subsets_temp.append(global_index)

            subsets_index.append(subsets_temp[:])
            subsets_temp = []

    return subsets_index


"Creates groups to classify all data for continuous features"
def createGroupsContinuous(segmentation,avg,std_dev):

    if segmentation == 2:

        groups = [-math.inf,avg,math.inf]

    elif segmentation == 4:

        groups = [-math.inf,avg-1*std_dev,avg,avg+1*std_dev,math.inf]

    elif segmentation == 6:

        groups = [-math.inf,avg-2*std_dev,avg-1*std_dev,avg,avg+1*std_dev,avg+2*std_dev,math.inf]

    return groups


"Attaches the group classification for all data to avoid recalculations of the same data while constructing the tree"
def attachTotalClassification(input_data,features,segmentation):

    "cno"
    avg_cno = features['cno'][0]
    std_dev_cno = features['cno'][1]
    groups_cno = createGroupsContinuous(segmentation,avg_cno,std_dev_cno)

    "lsq_residual"
    avg_lsq_residual = features['lsq_residual'][0]
    std_dev_lsq_residual = features['lsq_residual'][1]
    groups_lsq_residual = createGroupsContinuous(segmentation,avg_lsq_residual,std_dev_lsq_residual)

    "elevation"
    avg_elevation = features['elevation'][0]
    std_dev_elevation = features['elevation'][1]
    groups_elevation = createGroupsContinuous(segmentation,avg_elevation,std_dev_elevation)

    "azimuth"
    avg_azimuth = features['azimuth'][0]
    std_dev_azimuth = features['azimuth'][1]
    groups_azimuth = createGroupsContinuous(segmentation,avg_azimuth,std_dev_azimuth)

    "pdop"
    avg_pdop = features['pdop'][0]
    std_dev_pdop = features['pdop'][1]
    groups_pdop = createGroupsContinuous(segmentation,avg_pdop,std_dev_pdop)

    "ndop"
    avg_ndop = features['ndop'][0]
    std_dev_ndop = features['ndop'][1]
    groups_ndop = createGroupsContinuous(segmentation,avg_ndop,std_dev_ndop)

    "nr_used_measurements"
    groups_nr_used_measurements = features['nr_used_measurements']

    "constellation"
    groups_constellation = features['constellation']

    "tracking_type"
    groups_tracking_type = features['tracking_type']

    "multipath"
    groups_multipath = features['multipath']

    "raim"
    groups_raim = features['raim']

    "cycle_slip"
    groups_cycle_slip = features['cycle_slip']

    "difference_ENU_e"
    avg_difference_ENU_e = features['difference_ENU_e'][0]
    std_dev_difference_ENU_e = features['difference_ENU_e'][1]
    groups_difference_ENU_e = createGroupsContinuous(segmentation,avg_difference_ENU_e,std_dev_difference_ENU_e)

    "difference_ENU_n"
    avg_difference_ENU_n = features['difference_ENU_n'][0]
    std_dev_difference_ENU_n = features['difference_ENU_n'][1]
    groups_difference_ENU_n = createGroupsContinuous(segmentation,avg_difference_ENU_n,std_dev_difference_ENU_n)

    "difference_ENU_u"
    avg_difference_ENU_u = features['difference_ENU_u'][0]
    std_dev_difference_ENU_u = features['difference_ENU_u'][1]
    groups_difference_ENU_u = createGroupsContinuous(segmentation,avg_difference_ENU_u,std_dev_difference_ENU_u)

    "innovation_ENU_e"
    avg_innovation_ENU_e = features['innovation_ENU_e'][0]
    std_dev_innovation_ENU_e = features['innovation_ENU_e'][1]
    groups_innovation_ENU_e = createGroupsContinuous(segmentation,avg_innovation_ENU_e,std_dev_innovation_ENU_e)

    "innovation_ENU_n"
    avg_innovation_ENU_n = features['innovation_ENU_n'][0]
    std_dev_innovation_ENU_n = features['innovation_ENU_n'][1]
    groups_innovation_ENU_n = createGroupsContinuous(segmentation,avg_innovation_ENU_n,std_dev_innovation_ENU_n)

    "innovation_ENU_u"
    avg_innovation_ENU_u = features['innovation_ENU_u'][0]
    std_dev_innovation_ENU_u = features['innovation_ENU_u'][1]
    groups_innovation_ENU_u = createGroupsContinuous(segmentation,avg_innovation_ENU_u,std_dev_innovation_ENU_u)

    "sky"
    avg_sky = features['sky'][0]
    std_dev_sky = features['sky'][1]
    groups_sky = createGroupsContinuous(segmentation,avg_sky,std_dev_sky)

    global_index = 0

    for i in input_data:

        "cno"
        single_avg_cno,single_std_dev_cno = calculateStatistics(i['used_satellites'],'cno')

        "lsq_residual"
        single_avg_lsq_residual,single_std_dev_lsq_residual = calculateStatistics(i['used_satellites'],'lsq_residual')

        "elevation"
        single_avg_elevation,single_std_dev_elevation = calculateStatistics(i['used_satellites'],'elevation')

        "azimuth"
        single_avg_azimuth,single_std_dev_azimuth = calculateStatistics(i['used_satellites'],'azimuth')

        "pdop"
        single_avg_pdop = i['pdop']

        "ndop"
        single_avg_ndop = i['ndop']

        "nr_used_measurements"
        single_nr_used_measurements = i['nr_used_measurements']

        "constellation"
        single_constellation = mostCommonPerFeature(i['used_satellites'],'constellation')

        "tracking_type"
        single_tracking_type = mostCommonPerFeature(i['used_satellites'],'tracking_type')

        "multipath"
        single_multipath = mostCommonPerFeature(i['used_satellites'],'multipath')

        "raim"
        single_raim = mostCommonPerFeature(i['used_satellites'],'raim')

        "cycle_slip"
        single_cycle_slip = mostCommonPerFeature(i['used_satellites'],'cycle_slip')

        "difference_ENU_e"
        single_avg_difference_ENU_e = i['ekf_info']['difference_ENU']['e']

        "difference_ENU_n"
        single_avg_difference_ENU_n = i['ekf_info']['difference_ENU']['n']

        "difference_ENU_u"
        single_avg_difference_ENU_u = i['ekf_info']['difference_ENU']['u']

        "innovation_ENU_e"
        single_avg_innovation_ENU_e = i['ekf_info']['innovation_ENU']['e']

        "innovation_ENU_n"
        single_avg_innovation_ENU_n = i['ekf_info']['innovation_ENU']['n']

        "innovation_ENU_u"
        single_avg_innovation_ENU_u = i['ekf_info']['innovation_ENU']['u']

        "sky"
        single_avg_sky = i['sky']

        "Adding global identifier"
        i['global'] = global_index
        global_index += 1

        for j in range(0,len(groups_cno)-1):

            inferior_cno = groups_cno[j]
            superior_cno = groups_cno[j+1]

            if single_avg_cno >= inferior_cno and single_avg_cno <= superior_cno:

                i['cno_group'] = j
                break

        for j in range(0,len(groups_lsq_residual)-1):

            inferior_lsq_residual = groups_lsq_residual[j]
            superior_lsq_residual = groups_lsq_residual[j+1]

            if single_avg_lsq_residual >= inferior_lsq_residual and single_avg_lsq_residual <= superior_lsq_residual:

                i['lsq_residual_group'] = j
                break

        for j in range(0,len(groups_elevation)-1):

            inferior_elevation = groups_elevation[j]
            superior_elevation = groups_elevation[j+1]

            if single_avg_elevation >= inferior_elevation and single_avg_elevation <= superior_elevation:

                i['elevation_group'] = j
                break

        for j in range(0,len(groups_azimuth)-1):

            inferior_azimuth = groups_azimuth[j]
            superior_azimuth = groups_azimuth[j+1]

            if single_avg_azimuth >= inferior_azimuth and single_avg_azimuth <= superior_azimuth:

                i['azimuth_group'] = j
                break

        for j in range(0,len(groups_pdop)-1):

            inferior_pdop = groups_pdop[j]
            superior_pdop = groups_pdop[j+1]

            if single_avg_pdop >= inferior_pdop and single_avg_pdop <= superior_pdop:

                i['pdop_group'] = j
                break

        for j in range(0,len(groups_ndop)-1):

            inferior_ndop = groups_ndop[j]
            superior_ndop = groups_ndop[j+1]

            if single_avg_ndop >= inferior_ndop and single_avg_ndop <= superior_ndop:

                i['ndop_group'] = j
                break

        for j in range(0,len(groups_nr_used_measurements)):

            if single_nr_used_measurements == groups_nr_used_measurements[j]:

                i['nr_used_measurements_group'] = j
                break

        for j in range(0,len(groups_constellation)):

            if single_constellation == groups_constellation[j]:

                i['constellation_group'] = j
                break

        for j in range(0,len(groups_tracking_type)):

            if single_tracking_type == groups_tracking_type[j]:

                i['tracking_type_group'] = j
                break

        for j in range(0,len(groups_multipath)):

            if single_multipath == groups_multipath[j]:

                i['multipath_group'] = j
                break

        for j in range(0,len(groups_raim)):

            if single_raim == groups_raim[j]:

                i['raim_group'] = j
                break

        for j in range(0,len(groups_cycle_slip)):

            if single_cycle_slip == groups_cycle_slip[j]:

                i['cycle_slip_group'] = j
                break

        for j in range(0,len(groups_difference_ENU_e)-1):

            inferior_difference_ENU_e = groups_difference_ENU_e[j]
            superior_difference_ENU_e = groups_difference_ENU_e[j+1]

            if single_avg_difference_ENU_e >= inferior_difference_ENU_e and single_avg_difference_ENU_e <= superior_difference_ENU_e:

                i['difference_ENU_e_group'] = j
                break

        for j in range(0,len(groups_difference_ENU_n)-1):

            inferior_difference_ENU_n = groups_difference_ENU_n[j]
            superior_difference_ENU_n = groups_difference_ENU_n[j+1]

            if single_avg_difference_ENU_n >= inferior_difference_ENU_n and single_avg_difference_ENU_n <= superior_difference_ENU_n:

                i['difference_ENU_n_group'] = j
                break

        for j in range(0,len(groups_difference_ENU_u)-1):

            inferior_difference_ENU_u = groups_difference_ENU_u[j]
            superior_difference_ENU_u = groups_difference_ENU_u[j+1]

            if single_avg_difference_ENU_u >= inferior_difference_ENU_u and single_avg_difference_ENU_u <= superior_difference_ENU_u:

                i['difference_ENU_u_group'] = j
                break

        for j in range(0,len(groups_innovation_ENU_e)-1):

            inferior_innovation_ENU_e = groups_innovation_ENU_e[j]
            superior_innovation_ENU_e = groups_innovation_ENU_e[j+1]

            if single_avg_innovation_ENU_e >= inferior_innovation_ENU_e and single_avg_innovation_ENU_e <= superior_innovation_ENU_e:

                i['innovation_ENU_e_group'] = j
                break

        for j in range(0,len(groups_innovation_ENU_n)-1):

            inferior_innovation_ENU_n = groups_innovation_ENU_n[j]
            superior_innovation_ENU_n = groups_innovation_ENU_n[j+1]

            if single_avg_innovation_ENU_n >= inferior_innovation_ENU_n and single_avg_innovation_ENU_n <= superior_innovation_ENU_n:

                i['innovation_ENU_n_group'] = j
                break

        for j in range(0,len(groups_innovation_ENU_u)-1):

            inferior_innovation_ENU_u = groups_innovation_ENU_u[j]
            superior_innovation_ENU_u = groups_innovation_ENU_u[j+1]

            if single_avg_innovation_ENU_u >= inferior_innovation_ENU_u and single_avg_innovation_ENU_u <= superior_innovation_ENU_u:

                i['innovation_ENU_u_group'] = j
                break

        for j in range(0,len(groups_sky)-1):

            inferior_sky = groups_sky[j]
            superior_sky = groups_sky[j+1]

            if single_avg_sky >= inferior_sky and single_avg_sky <= superior_sky:

                i['sky_group'] = j
                break

    return input_data


"General calculate statistics method for satellites: used for CNO, lsq_residual, elevation, azimuth"
def calculateStatisticsSatellites(dataset,string):

    single_result = list()
    result = list()

    for x in dataset:

        [single_result.append(i[string]) for i in x['used_satellites']]
        avg_single_result = np.average(single_result)
        result.append(avg_single_result)
        single_result = []

    average = np.average(result)
    std_dev = np.std(result)

    return average,std_dev


"General calculate statistics method for general: used for PDOP, NDOP"
def calculateStatistics(dataset,string):

    result = list()
    [result.append(x[string]) for x in dataset]
    average = np.average(result)
    std_dev = np.std(result)

    return average,std_dev


"General calculate statistics method for general: used for EKF information"
def calculateStatisticsEkf(dataset,string):

    coordinate = string[-1]
    string = string[0:-2]
    result = list()
    [result.append(x['ekf_info'][string][coordinate]) for x in dataset]
    average = np.average(result)
    std_dev = np.std(result)

    return average,std_dev


"General indicate labels method for general: used for number_satellites"
def indicateLabel(dataset,string):

    number_sat = list()
    [number_sat.append(x[string]) for x in dataset]
    number_sat_list = list(set(number_sat))

    return number_sat_list


"General indicate labels method for general: used for constellation, tracking, multipath, raim, cycle_slip"
def indicateLabelSatellites(dataset,string):

    result = list()
    [result.append(x['used_satellites'][i][string]) for x in dataset for i in range(0,len(x['used_satellites']))]
    result_list = list(set(result))

    return result_list


"Counts the number of satellites in single time input corresponding to every feature. Whatever is the most important feature, will determine the label"
def mostCommonPerFeature(dataset,feature):

    feature_hits = list()
    [feature_hits.append(i[feature]) for i in dataset]
    most_common = max(set(feature_hits), key=feature_hits.count)

    return most_common


"Calculate the gains for all of the attributes and obtain the highest value"
def highestGainFeature(dataset,full,segmentation,filter_errors,histo_original,divisions,features):

    highest = list()
    features_label = list()
    tried = str(findTried(dataset))

    if tried.find("cno") == -1:
        gain_cno = gain(dataset,'cno',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('cno')
        highest.append(gain_cno)

    if tried.find("lsq_residual") == -1:
        gain_lsq_residual = gain(dataset,'lsq_residual',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('lsq_residual')
        highest.append(gain_lsq_residual)

    if tried.find("elevation") == -1:
        gain_elevation = gain(dataset,'elevation',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('elevation')
        highest.append(gain_elevation)

    if tried.find("azimuth") == -1:
        gain_azimuth = gain(dataset,'azimuth',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('azimuth')
        highest.append(gain_azimuth)

    if tried.find("pdop") == -1:
        gain_pdop = gain(dataset,'pdop',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('pdop')
        highest.append(gain_pdop)

    if tried.find("ndop") == -1:
        gain_ndop = gain(dataset,'ndop',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('ndop')
        highest.append(gain_ndop)

    if tried.find("nr_used_measurements") == -1:
        gain_nr_used_measurements = gain(dataset,'nr_used_measurements',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('nr_used_measurements')
        highest.append(gain_nr_used_measurements)

    if tried.find("constellation") == -1:
        gain_constellation = gain(dataset,'constellation',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('constellation')
        highest.append(gain_constellation)

    if tried.find("tracking_type") == -1:
        gain_tracking_type = gain(dataset,'tracking_type',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('tracking_type')
        highest.append(gain_tracking_type)

    if tried.find("multipath") == -1:
        gain_multipath = gain(dataset,'multipath',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('multipath')
        highest.append(gain_multipath)

    if tried.find("raim") == -1:
        gain_raim = gain(dataset,'raim',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('raim')
        highest.append(gain_raim)

    if tried.find("cycle_slip") == -1:
        gain_cycle_slip = gain(dataset,'cycle_slip',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('cycle_slip')
        highest.append(gain_cycle_slip)

    if tried.find("difference_ENU_e") == -1:
        gain_difference_ENU_e = gain(dataset,'difference_ENU_e',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('difference_ENU_e')
        highest.append(gain_difference_ENU_e)

    if tried.find("difference_ENU_n") == -1:
        gain_difference_ENU_n = gain(dataset,'difference_ENU_n',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('difference_ENU_n')
        highest.append(gain_difference_ENU_n)

    if tried.find("difference_ENU_u") == -1:
        gain_difference_ENU_u = gain(dataset,'difference_ENU_u',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('difference_ENU_u')
        highest.append(gain_difference_ENU_u)

    if tried.find("innovation_ENU_e") == -1:
        gain_innovation_ENU_e = gain(dataset,'innovation_ENU_e',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('innovation_ENU_e')
        highest.append(gain_innovation_ENU_e)

    if tried.find("innovation_ENU_n") == -1:
        gain_innovation_ENU_n = gain(dataset,'innovation_ENU_n',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('innovation_ENU_n')
        highest.append(gain_innovation_ENU_n)

    if tried.find("innovation_ENU_u") == -1:
        gain_innovation_ENU_u = gain(dataset,'innovation_ENU_u',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('innovation_ENU_u')
        highest.append(gain_innovation_ENU_u)

    if tried.find("sky") == -1:
        gain_sky = gain(dataset,'sky',full,segmentation,filter_errors,histo_original,divisions,features)
        features_label.append('sky')
        highest.append(gain_sky)

    return features_label[highest.index(max(highest))]


"Recursively build the decision tree"
def buildDecisionTree(dataset,segmentation,filter_errors,histo_original,average_per_bin,divisions,features,max_depth = 3):

    if max_depth == 1:

        data_tree = addNodeInfo(dataset,0)
        data_tree = addLeafInfo(dataset,0)
        data_tree = addTriedInfo(dataset,"")

        return data_tree

    tree = buildDecisionTree(dataset,segmentation,filter_errors,histo_original,average_per_bin,divisions,features,max_depth - 1)
    tree = buildTreeLevel(tree,segmentation,filter_errors,histo_original,average_per_bin,divisions,features)

    return tree


"Build levels or branches of the tree:"
def buildTreeLevel(tree,segmentation,filter_errors,histo_original,average_per_bin,divisions,features):
    print("started tree level construction")
    start_level = time.time()
    data_tree = list()
    nodes = findNodes(tree)
    node_index = list()

    for node in nodes:

        [node_index.append(j) for j in range(0,len(tree)) if tree[j]['node'] == node and tree[j]['leaf'] == 0]

        leaf_verify = verifyEntropy(node_index,tree,average_per_bin,filter_errors,histo_original,divisions)
        node_dataset = getData(node_index,tree)

        if leaf_verify == 0:

            bestFeature = highestGainFeature(node_dataset,tree,segmentation,filter_errors,histo_original,divisions,features)
            subsets = selectByFeature(node_dataset,bestFeature,segmentation,features)
            data_tree = addNodeToData(node_dataset,subsets,tree)
            data_tree = addTriedToData(node_dataset,bestFeature,tree)
            node_dataset = []
            node_index = []

        else:

            data_tree = addLeafToData(node_dataset,leaf_verify,tree)
            node_dataset = []
            node_index = []
    end_level = time.time()
    diff_level = end_level - start_level
    print("Finished tree level construction execution time (s): "+str(diff_level))
    print("Finished tree level construction execution time (min): "+str(diff_level/60))
    return data_tree


"Add node and tree address for each node in a dataset"
def addNodeToData(dataset,subsets,tree):

    dataset_node = dataset[:]
    data_tree = tree[:]
    len_subsets = len(subsets)

    for i in range(0,len_subsets):

        index_i = subsets[i]

        for j in range(0,len(index_i)):

            index_ij = index_i[j]

            if isinstance(index_ij,dict):

                location_at_node = index_i.index(index_ij)

                if str(index_i[location_at_node]['node']) == "":

                    new_node = str(index_i[location_at_node]['node'])+str(i)


                else:

                    new_node = str(index_i[location_at_node]['node'])+"-"+str(i)

                data_tree[data_tree.index(index_ij)]['node'] = new_node

            else:

                location_at_node = dataset_node.index(data_tree[index_ij])

                if str(dataset_node[location_at_node]['node']) == "":

                    new_node = str(dataset_node[location_at_node]['node'])+str(i)

                else:

                    new_node = str(dataset_node[location_at_node]['node'])+"-"+str(i)

                data_tree[index_ij]['node'] = new_node

    return data_tree


"Add completed Leaf information once the node is done in classification"
def addLeafToData(dataset,parameter,tree):

    dataset_node = dataset[:]
    data_tree = tree[:]
    len_dataset = len(dataset)

    for i in range(0,len_dataset):

        data_tree[data_tree.index(dataset_node[i])]['leaf']=parameter

    return data_tree


"Adds the best features tried to each node"
def addTriedToData(dataset,parameter,tree):

    dataset_node = dataset[:]
    data_tree = tree[:]
    len_dataset = len(dataset)

    if dataset_node != []:

        if isinstance(dataset_node[0],dict):

            for i in range(0,len_dataset):

                new_tried = str(dataset_node[i]['tried'])+" "+parameter
                location_at_tree = data_tree.index(dataset_node[i])
                data_tree[location_at_tree]['tried']=new_tried

        else:

            for i in range(0,len_dataset):

                line = dataset_node[i]

                for j in range(0,len(line)):

                    new_tried = str(dataset_node[i][j]['tried'])+" "+parameter
                    location_at_tree = data_tree.index(dataset_node[i][j])
                    data_tree[location_at_tree]['tried']=new_tried

    return data_tree


"Get dataset from a list of indexes. Indexes can be organized in subsets"
def getData(indexes,tree):

    dataset = list()
    [dataset.append(tree[i]) for i in indexes]

    return dataset


"Get indexes of a dataset"
def getIndex(dataset_interest,dataset):

    index = list()
    [index.append(dataset.index(dataset_interest[i])) for i in range(0,len(dataset))]

    return index


"Verifies entropy level and returns the average value per bin in which a node has been alocated to. Receives indexes"
def verifyEntropy(indexes,tree,average_per_bin,filter_errors,histo_original,divisions):

    leaf_val = 0
    dataset = getData(indexes,tree)
    errors_set = getErrors(dataset,tree,filter_errors)
    entropy_node = calculateEntropy(errors_set,histo_original,divisions)

    if isinstance(entropy_node,dict):

            leaf_val = average_per_bin[entropy_node['location']]

    return leaf_val


"Creates an additional feature inside filter_input_data related to the nodes in the decision tree"
def addNodeInfo(dataset,parameter):

    for i in dataset:

        i['node']=parameter

    return dataset


"Creates an additional feature inside filter_input_data_node related to the leaf value in the decision tree"
def addLeafInfo(dataset,parameter):

    for i in dataset:

        i['leaf']=parameter

    return dataset


"Creates an additional feature inside filter_input_data_node_leaf related to the tried features in the decision tree"
def addTriedInfo(dataset,parameter):

    for i in dataset:

        i['tried']=0

    return dataset


"Finds different number of nodes in every iteration or call"
def findNodes(dataset):

    nodes = list()
    [nodes.append(i['node']) for i in dataset]
    nodes_set = set(nodes)

    return nodes_set


"Finds the features Tried for each node"
def findTried(dataset):

    tried_all = list()
    [tried_all.append(i['tried']) for i in dataset]
    tried_set = set(tried_all)

    return tried_set


"Finds all indexes in the tree corresponding to the denomination per node"
def findIndexPerNode(tree):

    node_index_subset = list()
    node_index = list()
    nodes = findNodes(tree)

    for node in nodes:

        [node_index_subset.append(i) for i in range(0,len(tree)) if tree[i]['node'] == node]

        subset = {
            'node':node,
            'index':node_index_subset
            }

        node_index.append(subset)
        subset = []
        node_index_subset = []

    return node_index


"Finds all indexes in the tree corresponding to the denomination per node"
def findTriedPerNode(tree):

    node_tried_2 = list()
    seen = set()

    for i in tree:

        node_i = i['node']
        tried_i = i['tried']

        sample = {
            'node':node_i,
            'features':tried_i
            }

        t = tuple(sample.items())

        if t not in seen:
            seen.add(t)
            node_tried_2.append(sample)

    return node_tried_2


"Finds all errors per node on the tree"
def findErrorsPerNode(nodes_index,tree,filter_errors):

    data_subset = list()
    error_subset = list()
    errors = list()

    for i in nodes_index:

        node = i['node']
        index_subset = i['index']
        data_subset = getData(index_subset,tree)
        error_subset = getErrors(data_subset,tree,filter_errors)

        sample = {
            'node':node,
            'errors':error_subset
            }

        errors.append(sample)

    return errors


"Finds all error fractions per node on the tree"
def findErrorFractionsPerNode(error_list,divisions):

    fraction_sample = list()
    fractions = list()

    for i in error_list:

        node = i['node']
        errors = i['errors']
        n,edges = np.histogram(errors,bins=divisions,density=False)
        n_total = sum(n)

        for j in range(0,len(n)):

            n_j = n[j]

            if n_total == 0:

                prob = 0

            else:

                prob = n_j/n_total

            fraction_sample.append(prob)

        fraction_subset = {
            'node':node,
            'fractions':fraction_sample
            }
        fraction_sample = []

        fractions.append(fraction_subset)
        fraction_subset = []

    return fractions


"Get the most important (highest) fraction for each node"
def findHighestFractionIndex(fractions):

    highest = -1
    node_high = list()

    for i in range(0,len(fractions)):

        node = fractions[i]['node']
        fraction_node = fractions[i]['fractions']
        highest = max(fraction_node)
        highest_index = fraction_node.index(highest)

        sample = {
            'node':node,
            'index':highest_index
            }

        node_high.append(sample)

    return node_high


"Find the average per bin label corresponding to the highest fraction of the leaves"
def findLabel(nodes_highest_frac_index,average_per_bin):

    label = list()

    for i in range(0,len(nodes_highest_frac_index)):

        node = nodes_highest_frac_index[i]['node']
        index_node = nodes_highest_frac_index[i]['index']
        label_node = average_per_bin[index_node]

        sample = {
            'node':node,
            'label':label_node
            }

        label.append(sample)

    return label


"Total classification error calculation"
def classificationPerformanceTrain(error_list,divisions):

    miss = 0
    correct = 0
    total = 0
    classification_error = 0

    for i in error_list:

        errors = i['errors']
        n,edges = np.histogram(errors,bins=divisions,density=False)
        n_total = sum(n)
        n_correct = max(n)
        n_miss = n_total - n_correct

        miss = miss + n_miss
        correct = correct + n_correct
        total = total + n_total

    classification_error = miss/total

    return classification_error


"Total classification error calculation for test"
def classificationPerformanceTest(error_list,divisions,nodes_labels_train,average_per_bin):

    miss = 0
    correct = 0
    total = 0
    classification_error = 0

    for i in error_list:

        found = 0
        node_test = i['node']
        node_errors = i['errors']
        node_errors_len = len(node_errors)
        n,edges = np.histogram(node_errors,bins=divisions,density=False)
        total_n = sum(n)

        for j in nodes_labels_train:

            node_train = j['node']

            if node_test == node_train:

                found = 1
                label_train_i = j['label']
                index_avg_per_bin = average_per_bin.index(label_train_i)
                n_at_label_train = n[index_avg_per_bin]
                correct = correct + n_at_label_train
                total = total + node_errors_len
                miss = miss + total_n - n_at_label_train

        if found == 0:

            max_n = max(n)
            correct = correct + max_n
            total = total + node_errors_len
            miss = miss + total_n - max_n

    classification_error = miss/total

    return classification_error


"indicates which feature per node is used for splitting"
def constructTree(node_tried):

    tree_list = list()

    for i in node_tried:

        node_i = str(i['node'])
        features_i = i['features']
        node_i_split = node_i.split("-")
        features_i_split = str.split(features_i)
        len_node_i_split = len(node_i_split) - 1
        nod = ""

        for j in range(0,len_node_i_split):

            if nod == "":

                nod = nod+str(node_i_split[j])

            else:

                nod = nod+"-"+str(node_i_split[j])

            feat = features_i_split[j+1]

            sample = {
                'node':nod,
                'feature':feat
                }

            if verifyNotInList(tree_list,sample):

                tree_list.append(sample)

    return tree_list


"Function to find if the node has already been saved in the tree list"
def verifyNotInList(tree_list,sample):

    var = True

    sample_node = sample['node']

    for i in tree_list:

        node = i['node']

        if node == sample_node:

            var = False

    return var


"Method to find the error of the Siblings in case the partition does not exist in tree labels"
def findSiblingsErrors (node,nodes_errors):

    parent_node = node[0:-1]
    siblingErrors = list()
    [siblingErrors.extend(i['errors']) for i in nodes_errors if str(i['node'])[0:-1] == parent_node]

    return siblingErrors


"Main method: Implements decision tree building methods"
def tree(filter_input_train,filter_output_train,filter_errors_train,input_test,output_test,features,divisor_bins,segmentation,test_fraction,tree_depth):

    "Create the divisions for the classes for the histogram and decision trees"
    divisions = createBins(filter_errors_train,divisor_bins)


    "Creates the error vector for each of the divisions in the errors list"
    list_bins = listPerBin(divisions,filter_errors_train)


    "Creates the average value per bin created based on the divisions created"
    average_per_bin = averagePerBin(list_bins)


    "Obtains the histogram based on the divisions created for the errors list"
    histo_original,edges = np.histogram(filter_errors_train,bins=divisions,density=False)


    "Attaches all group classification for all data in advance to accelerate SelectByFeature method"
    start_attach = time.time()
    filter_input_train = attachTotalClassification(filter_input_train,features,segmentation)
    end_attach = time.time()
    diff_attach = end_attach - start_attach
    print("Finished attaching all data in advance (s): "+str(diff_attach))
    print("Finished attaching all data in advance (min): "+str(diff_attach/60))

    "Gathers important information from the tree to be returned"
    data_in_tree = buildDecisionTree(filter_input_train,segmentation,filter_errors_train,histo_original,average_per_bin,divisions,features,tree_depth)

    start_total = time.time()
    start_tried = time.time()
    node_tried = findTriedPerNode(data_in_tree)
    end_tried = time.time()
    diff_tried = end_tried - start_tried
    print("Finished node tried (s): "+str(diff_tried))
    print("Finished node tried (min): "+str(diff_tried/60))

    start_classifier = time.time()
    tree_classifier = constructTree(node_tried)
    end_classifier = time.time()
    diff_classifier = end_classifier - start_classifier
    print("Finished classifier (s): "+str(diff_classifier))
    print("Finished classifier (min): "+str(diff_classifier/60))

    start_index = time.time()
    nodes_index = findIndexPerNode(data_in_tree)
    end_index = time.time()
    diff_index = end_index - start_index
    print("Finished node index (s): "+str(diff_index))
    print("Finished node index (min): "+str(diff_index/60))

    start_errors = time.time()
    nodes_errors = findErrorsPerNode(nodes_index,data_in_tree,filter_errors_train)
    end_errors = time.time()
    diff_errors = end_errors - start_errors
    print("Finished node errors (s): "+str(diff_errors))
    print("Finished node errors (min): "+str(diff_errors/60))

    start_fractions = time.time()
    nodes_fractions = findErrorFractionsPerNode(nodes_errors,divisions)
    end_fractions = time.time()
    diff_fractions = end_fractions - start_fractions
    print("Finished node fractions (s): "+str(diff_fractions))
    print("Finished node fractions (min): "+str(diff_fractions/60))

    start_high = time.time()
    nodes_high_fraction = findHighestFractionIndex(nodes_fractions)
    end_high = time.time()
    diff_high = end_high - start_high
    print("Finished node high (s): "+str(diff_high))
    print("Finished node high (min): "+str(diff_high/60))

    start_labels = time.time()
    nodes_labels = findLabel(nodes_high_fraction,average_per_bin)
    end_labels = time.time()
    diff_labels = end_labels - start_labels
    print("Finished node labels (s): "+str(diff_labels))
    print("Finished node labels (min): "+str(diff_labels/60))

    start_classification = time.time()
    classification_error = classificationPerformanceTrain(nodes_errors,divisions)
    end_classification = time.time()
    diff_classification = end_classification - start_classification
    print("Finished node classification (s): "+str(diff_classification))
    print("Finished node classification (min): "+str(diff_classification/60))

    end_total = time.time()
    diff_total = end_total - start_total
    print("Finished all information (s): "+str(diff_total))
    print("Finished all information (min): "+str(diff_total/60))

    return classification_error,tree_classifier,nodes_labels,nodes_errors,input_test,output_test,filter_input_train,filter_output_train,divisions,average_per_bin,features,data_in_tree


"--------------------------------------------------------------------------------------"
"-------------------------------------tree tester--------------------------------------"
"--------------------------------------------------------------------------------------"


"Method to test a tree's performance on the input data selected"
def testTree(input_test,output_test,divisor_bins,segmentation,tree_classifier_train,nodes_labels_train,divisions,average_per_bin,features,tree_depth):
    print("Starting testing tree built")
    start_test = time.time()

    "Reads the input and output data"
    input_data = input_test
    output_data = output_test

    "Automatically calculates errors only from the output vector"
    errors = list()

    for i in range(0,len(output_data)):

        errors.append(output_data[i]['error'])

    filter_input_data = input_data
    filter_errors = errors


    "Obtains the histogram based on the divisions created for the errors list"
    histo_original,edges = np.histogram(filter_errors,bins=divisions,density=False)

    filter_input_data = addNodeInfo(filter_input_data,0)
    filter_input_data = addLeafInfo(filter_input_data,0)
    filter_input_data = addTriedInfo(filter_input_data,"")

    def treeExecution(tree_data,errors,segmentation,features,divisions,tree_depth,tree_classifier_train,nodes_labels_train,average_per_bin):

        old_tree = tree_data

        for i in range(0,tree_depth-1):

            tree = splitNodeByFeature(old_tree,segmentation,features,tree_classifier_train)
            old_tree = tree

        nodes_index = findIndexPerNode(tree)
        nodes_errors = findErrorsPerNode(nodes_index,tree,errors)
        nodes_labels = labelTestNodes(tree,nodes_labels_train,nodes_errors)
        classification_error = classificationPerformanceTest(nodes_errors,divisions,nodes_labels_train,average_per_bin)

        return tree,classification_error,nodes_labels,nodes_errors


    def splitNodeByFeature(dataset,segmentation,features,tree_classifier):
        "Splits the current node into subnodes or branch out process according to the feature"
        nodes = findNodes(dataset)
        data_per_node = list()
        subset_data = list()

        for i in nodes:

            for j in dataset:

                if j['node'] == i:

                    data_per_node.append(j)

            feature = findFeaturePerNode(i,tree_classifier)

            if feature == "Not found":

                data_per_node = []
                continue

            if feature == 'cno' or feature == 'lsq_residual' or feature == 'elevation' or feature == 'azimuth':

                selector = 0
                avg = features[feature][0]
                std_dev = features[feature][1]
                groups = segments(avg,std_dev,segmentation)
                subset_data = setDataIntoSplitNodesContinuous(groups,data_per_node,selector,feature)

            elif feature == 'pdop' or feature == 'ndop' or feature == 'sky':

                selector = 1
                avg = features[feature][0]
                std_dev = features[feature][1]
                groups = segments(avg,std_dev,segmentation)
                subset_data = setDataIntoSplitNodesContinuous(groups,data_per_node,selector,feature)

            elif feature == 'nr_used_measurements':

                selector = 3
                subset_data = setDataIntoSplitNodesDiscrete(data_per_node,selector,feature)

            elif feature == 'constellation' or feature == 'tracking_type' or feature == 'multipath' or feature == 'raim' or feature == 'cycle_slip':

                selector = 4
                subset_data = setDataIntoSplitNodesDiscrete(data_per_node,selector,feature)

            else:

                selector = 5
                avg = features[feature][0]
                std_dev = features[feature][1]
                groups = segments(avg,std_dev,segmentation)
                subset_data = setDataIntoSplitNodesContinuous(groups,data_per_node,selector,feature)

            dataset = addNodeToData(subset_data,subset_data,dataset)
            dataset = addTriedToData(subset_data,feature,dataset)
            data_per_node = []

        return dataset

    def setDataIntoSplitNodesContinuous(groups,subset,selector,feature):
        "Splits a node into its children nodes based on the grouping and statistics of the node"
        subsets_temp = list()
        subsets = list()

        for i in range(0,len(groups)-1):

            inferior = groups[i]
            superior = groups[i+1]

            for j in subset:

                if selector == 0:

                    avg,std_dev = calculateStatistics(j['used_satellites'],feature)

                elif selector == 1:

                    avg = j[feature]

                elif selector == 5:

                    sub_string = feature[0:-2]
                    coordinate = feature[-1]
                    avg = j['ekf_info'][sub_string][coordinate]

                if avg >= inferior and avg <= superior:

                    subsets_temp.append(j)

            subsets.append(subsets_temp[:])
            subsets_temp = []

        return subsets


    def setDataIntoSplitNodesDiscrete(subset,selector,feature):
        "Splits a node into its children nodes based on the grouping and statistics of the node"
        subsets_temp = list()
        subsets = list()
        groups = features[feature]

        for i in groups:

            group_label = i

            for j in subset:

                if selector == 3:

                    info = j['nr_used_measurements']

                    if info == group_label:

                        subsets_temp.append(j)

                elif selector == 4:

                    most_common = mostCommonPerFeature(j['used_satellites'],feature)

                    if most_common == group_label:

                        subsets_temp.append(j)

            subsets.append(subsets_temp[:])
            subsets_temp = []

        return subsets


    def segments(avg,std_dev,segmentation):
        "Segments the possible range of values into average and standard deviation dependent segments"
        if segmentation == 2:

            groups = [-math.inf,avg,math.inf]

        elif segmentation == 4:

            groups = [-math.inf,avg-1*std_dev,avg,avg+1*std_dev,math.inf]

        elif segmentation == 6:

            groups = [-math.inf,avg-2*std_dev,avg-1*std_dev,avg,avg+1*std_dev,avg+2*std_dev,math.inf]

        return groups


    def findFeaturePerNode(current_node,tree_classifier):
        "Finds nodes feature used to branch the current node"

        feature = "Not found"

        for i in tree_classifier:

            node = i['node']

            if str(current_node) == node:

                feature = i['feature']

        return feature


    def findNodeLabel(node,node_label):

        min_labels = list()
        max_labels = list()
        found = 0
        j = 1

        while found == 0:

            parent_node = node[0:-2*j]

            for i in node_label:

                if i['node'][0:-2*j] == parent_node:

                    found = 1
                    min_labels.append(i['label']['min'])
                    max_labels.append(i['label']['max'])

            j += 1

        min_average = np.average(min_labels)
        max_average = np.average(max_labels)

        return min_average,max_average


    def labelTestNodes(tree,nodes_label,nodes_errors):

        nodes = findNodes(tree)
        nodes_labels_test = list()

        for i in nodes:

            found = 0

            for j in nodes_label:

                if i == j['node']:

                    found = 1
                    label_node = j['label']
                    sample = {
                        'node':i,
                        'label':label_node
                        }
                    nodes_labels_test.append(sample)

            if found == 0:

                for k in nodes_errors:

                    if i == k['node']:


                        min_val,max_val = findNodeLabel(i,nodes_label)
                        sample = {
                        'node':i,
                        'label':{
                            'min':min_val,
                            'max':max_val
                                }
                        }
                        nodes_labels_test.append(sample)
                        min_val = 0
                        max_val = 0

        return nodes_labels_test

    tree,classification_error,nodes_labels,nodes_errors = treeExecution(filter_input_data,filter_errors,segmentation,features,divisions,tree_depth,tree_classifier_train,nodes_labels_train,average_per_bin)
    end_test = time.time()
    diff_test = end_test - start_test
    print("Finished tree testing (s): "+str(diff_test))
    print("Finished tree testing (min): "+str(diff_test/60))
    return classification_error,nodes_labels,nodes_errors,tree
