import os 
import numpy as np 
import pandas as pd 



def load_data(tmp_folder,filename=None):
    
    if filename[-3:] == 'npy':
        var= np.load(os.path.join(tmp_folder,filename),allow_pickle=True)
    else:
        var= pd.read_csv(os.path.join(tmp_folder,filename),index_col=0)
        var=var.sort_values('index')
        
    return var

def merge_data (start_dir,list_of_animals,filename=None):
    
    '''
    
    function to merge  animal specific files. call with:
    
    start_dir = r'F:\Reduced_Datasets_v1\animal_data\wt'
    
    filenames:
    binned_data_name = 'VPL_pop_250.0ms_data.npy'
    temp_traces_name = 'VPL_pop_amp_traces.npy'
    metadata_name = 'VPL_pop_metadata.csv'
    raster_name = 'VPL_pop_rasters.npy'
    sdf_name = 'VPL_pop_sdf.npy'
    popstats_name = 'VPL_pop_stats.csv'
    
    '''
    data_files = []
    
    for i in range(len(list_of_animals)):
        
        tmp_folder = os.path.join(start_dir,list_of_animals[i]) 
            
        data_tmp = load_data(tmp_folder,filename)
        data_files.append(data_tmp)
        
    if filename[-3:] == 'npy':
        data = np.concatenate(data_files)
    else:
        data= pd.concat(data_files)
         
        
    return data