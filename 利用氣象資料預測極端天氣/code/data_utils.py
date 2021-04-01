import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import glob
from zipfile import *
import torch.utils.data as Data
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import math

def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_files(station_id,directory_path,output_file_name,extension):
    delete_file('{}/{}'.format(directory_path,output_file_name))
    
    all_filenames = [i for i in glob.glob('{}/{}*.{}'.format(directory_path,station_id,extension))]
    print('Number of files: {}'.format(len(all_filenames)))

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

    #export to csv
    combined_csv.to_csv('{}/{}'.format(directory_path,output_file_name), index=False, encoding='utf-8-sig')

def remove_null_column(row, column_name):
    """
    Remove null column value
    """
    if(math.isnan(row[column_name])):
        return 0
    else:
        return row[column_name]

def generate_time_shift_columns(row,time_shift):
    """
    Observation data to join itself to have the previous TIME_SHIFT hours' data as new columns
    The TIME_SHIFT's columns will be used as the encoder input
    """
    timespan = timedelta(hours=time_shift)
    yyyymmddhh = str(row['yyyymmddhh'])[0:10]

    if(yyyymmddhh[8:10]=='24'):
        calculated_date = datetime.strptime(yyyymmddhh[0:8],'%Y%m%d') - timespan
    else:
        calculated_date = datetime.strptime(yyyymmddhh,'%Y%m%d%H') - timespan
    final_time = calculated_date.strftime("%Y-%m-%d %H:%M:%S") + " UTC"
    return final_time

def arrange_abnormal_data(row,column_name,updated_value):
    if(row[column_name]<=-9991):
        return updated_value
    else:
        return row[column_name]

def join(left_table, right_table, output_file, debug):
    # For debug purpose
    if(debug):
        left_table.to_csv('left_table.csv')
        right_table.to_csv('right_table.csv')

    # res = pd.merge(left_table,right_table,on=['start_time'],how='outer')
    res = pd.merge(left_table,right_table,on=['shifted_start_time'],how='inner')
    res2 = res.sort_values(['issue_time','yyyymmddhh'], ascending=[True,True])

    if(os.path.exists(output_file)):
        os.remove(output_file)
    res2.to_csv(output_file, index=False)
    return res2

def standard_normalization(row,column_name,index,mean_series,std_series):
    return((row[column_name]-mean_series[index])/std_series[index])


def get_io_data(keys,df,input_columns,output_columns,squeeze_output_flag):
    """
    Input format (batch_size, seq_len, input_size)
    Output format (batch_size, seq_len)
    batch_size will be dealed in the Data.DataLoader so we could ignore here
    Input format (seq_len, input_size)
    Output format (seq_len)
    """
    inputs = []
    outputs = []
    for key in keys:
        selected_group = df.get_group(key).reset_index(drop=True)
        if(len(selected_group)==193):
            inputs.append(selected_group[input_columns].values)
            ## TODO: LSTM doesn't need squeeze
            # output = np.squeeze(selected_group[output_columns].values, axis=1)
            if squeeze_output_flag:
                output = np.squeeze(selected_group[output_columns].values, axis=1)
                outputs.append(output)
            else:
                output = selected_group[output_columns].values
                outputs.append(output)
    return (inputs,outputs)

def merge_plotted_data(plotted_data_list):
    """
    Concatenate the data list for the plotted images to single numpy array
    param plotted_data_list: the data list for the plotted images
    """
    plotted_data = []
    for item in plotted_data_list:
        item = torch.Tensor.cpu(item).detach().numpy()
        if plotted_data == []:
            # plotted_data = torch.Tensor.cpu(item).detach().numpy()
            # plotted_data = np.append(a,item)
            plotted_data = item
        else:
            # plotted_data ＝ np.concatenate((plotted_data, item), axis=2)
            plotted_data = np.append(plotted_data,item,axis=0)
    
    return plotted_data

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def mse_4th_loss(output, target):
    loss = torch.mean((output - target)**4)
    return loss

