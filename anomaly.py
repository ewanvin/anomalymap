from datetime import datetime
import argparse
import os 
import sys 
import logging.handlers
import xarray as xr
import yaml 
import numpy as np 
import dask
from dask.diagnostics import ProgressBar, ResourceProfiler
import pandas as pd 
import subprocess
import glob


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg",dest="cfgfile",
                        help="Configuration file", required=True)
    
    args = parser.parse_args()

    if args.cfgfile is False:
        parser.print_help()
        parser.exit()

    return args

def parse_cfg(cfgfile):
    # Read config file
    print('Reading', cfgfile)
    with open(cfgfile, 'r') as ymlfile:
        cfgstr = yaml.full_load(ymlfile)

    return cfgstr

def initialise_logger(outputfile):
    mylog = logging.getLogger()
    mylog.setLevel(logging.INFO)
    myformat = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.handlers.TimedRotatingFileHandler(outputfile, when='w0', interval=1, backupCount=7)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(myformat)
    mylog.addHandler(file_handler)
    
    return mylog

def calculate_anomaly(dataset,cfgstr,mylog):
    # Get variable name from cfgstr
    var_name = cfgstr['input']['variable']
    
    # Extract endpoint time
    endpoint = dataset.time[-1].values
    print('Endpoint:',endpoint)

    # Convert to pandas Timestamp
    timestamp = pd.Timestamp(endpoint)

    # Subtract number of years
    span_1 = timestamp - pd.DateOffset(years=cfgstr['input']['period'])
    span_2 = timestamp - pd.DateOffset(years=(cfgstr['input']['period'])*2)
    print('Span 1:',span_1)
    print('Span 2:',span_2)

    start_year = pd.to_datetime(dataset.time[0].values).year 
    print('Start year:',start_year)

    if span_2.year < start_year:
        raise ValueError('The choice of period superseeds the length of data. Please select a smaller period range.')
    else:
        print('Period selection is within time period.')

    # Create date strings 
    date_string_1 = span_1.strftime('%Y-%m-%d')
    date_string_2 = span_2.strftime('%Y-%m-%d')

    #if 'data_endpoint' in cfgstr['input'] and cfgstr['input']['data_endpoint']:
    if 'input' in cfgstr and cfgstr['input'] is not None:
        if 'cropped' in cfgstr['input']:
            if cfgstr['input']['cropped'] == 'Yes':
                print('Cropping data...')
            
                # Slicing Dataset
                cropped = dataset.sel(xc=slice(0, -3250))
                
                # Closing Dataset
                dataset.close()

                cropped_nbytes_gb = cropped.nbytes / (1024*1024*1024)
                
                print('Cropped Dataset size in GB:', cropped_nbytes_gb)
                mylog.info('Cropped Dataset size in GB: %s', cropped_nbytes_gb)
                
                # Calculate the mean for the entire dataset
                mean = cropped[var_name].mean(dim='time')
                            
                # Select the time range for each period
                mylog.info('Selecting time range for each period...')
                period_1 = cropped[var_name].sel(time=slice(f'{date_string_1}', f'{str(dataset.time[-1].values)}'))
                period_2 = cropped[var_name].sel(time=slice(f'{date_string_2}', f'{date_string_1}'))
                mylog.info('Time range selection done.')

                # Calculate the anomalies for each period
                anom_period_1 = period_1.mean(dim='time') - mean
                anom_period_1 = xr.where(np.isnan(anom_period_1), np.nan, anom_period_1)
                
                anom_period_2 = period_2.mean(dim='time') - mean
                anom_period_2 = xr.where(np.isnan(anom_period_2), np.nan, anom_period_2)
            
                mylog.info('Computing anoamly periods 1 and 2...')
                # Compute all results at once
                anom_period_1 = dask.compute(anom_period_1, scheduler='synchronous')[0]
                anom_period_2 = dask.compute(anom_period_2, scheduler='synchronous')[0]
                mylog.info('Computation of anoamly periods 1 and 2 finished')

                return cropped,anom_period_1, anom_period_2 
        
            elif cfgstr['input']['cropped'] == 'No':
                print('Data does not need to be cropped.')

                if len(dataset.dims) > 2:
                    print('The dataset has more than 4 dimensions.')
                    # Compute mean of the Dataset
                    mean = dataset[var_name].mean(dim='time')
                    
                    mylog.info('Selecting time range for each period...')
                    period_1 = dataset[var_name].sel(time=slice(f'{date_string_1}', f'{str(dataset.time[-1].values)}'))
                    period_2 = dataset[var_name].sel(time=slice(f'{date_string_2}', f'{date_string_1}'))
                    mylog.info('Time range selection done.')

                    mylog.info('Calculating anomalies for each period...')
                    anom_period_1 = period_1.mean(dim='time') - mean
                    anom_period_1 = xr.where(np.isnan(anom_period_1), np.nan, anom_period_1)
                    
                    anom_period_2 = period_2.mean(dim='time') - mean
                    anom_period_2 = xr.where(np.isnan(anom_period_2), np.nan, anom_period_2)
                    mylog.info('Calculation of anomalies completed.')


                    return anom_period_1, anom_period_2

                elif len(dataset.dims) == 2:
                    print('The dataset has 2 dimensions.')
                    
                    # Use variable name to select data from dataset
                    period_1 = dataset[var_name].sel(time=slice(f'{date_string_1}', f'{str(dataset.time[-1].values)}'))
                    period_2 = dataset[var_name].sel(time=slice(f'{date_string_2}', f'{date_string_1}'))
                    
                    # Group by month and calculate mean for each period
                    gb_period_1 = period_1.groupby('time.day')
                    gb_period_2 = period_2.groupby('time.day')
                    
                    clim_period_1 = gb_period_1.mean(dim='time')
                    clim_period_2 = gb_period_2.mean(dim='time')
                    
                    # Calculate anomalies for each period
                    anom_period_1 = gb_period_1 - clim_period_1
                    anom_period_2 = gb_period_2 - clim_period_2

                    print('Size of anomaly period 1', anom_period_1.nbytes / (1024*1024), 'MB')
                    print('Size of anomaly period 2', anom_period_2.nbytes / (1024*1024), 'MB')

                    return anom_period_1, anom_period_2
            
            elif cfgstr['input']['cropped'] == '':
                print('cropped key is empty.')

                return 
            
        else:
            print('cropped is not defined in the configuration file.')

            return    

    else: 
        print("'input' key is not defined in the configuration file") 

        return   


# Get dimensions using ncdump
def get_dimensions(file_pattern):
    filenames = glob.glob(file_pattern)
    #print(f"Files: {filenames}")  
    dimensions = {}
    for filename in filenames:
        result = subprocess.run(['ncdump', '-h', filename], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        lines = output.split('\n')
        for line in lines:
            if line.startswith('\t') and '=' in line and ':' not in line:
                parts = line.split(' = ')
                if len(parts) == 2 and parts[1].endswith(' ;'):
                    key = parts[0].lstrip()  # Remove leading whitespace
                    dimensions[key] = int(parts[1].rstrip(' ;'))
    print(f'Dimensions found in the dataset: {dimensions}') 
    return dimensions


def write_to_netcdf(dataset, cfgstr, period):
    filename = cfgstr['output']['datafile'].replace('.nc',f'_anom_period_{period}.nc')
    
    dims = list(dataset.sizes.keys())
    #print('dims:',dims)
    chunksizes = []
    encoding_dict = {cfgstr['input']['variable']: {'zlib': True, 'complevel': 9, 'dtype': 'float64'}}

    # Calculate chunksizes for each dimension
    for dim in dims:
        coord_len = dataset.sizes[dim]
        #print(f'coordinate {dim}:', coord_len)
        chunksizes.append(int(coord_len / 15))

    # Add chunksizes to encoding dict if more than one dimension
    if len(dims) > 1:
        encoding_dict[cfgstr['input']['variable']]['chunksizes'] = tuple(chunksizes)
    
    #print('chunksizes:',chunksizes)

    # Chunk handling for large datasets 
    if dataset.nbytes > 5000000000:
        dataset.to_netcdf(filename, encoding=encoding_dict)
    else:
        dataset.to_netcdf(filename)


def main():

    args = parse_arguments()
    cfgstr = parse_cfg(args.cfgfile)
    mylog = initialise_logger(cfgstr['output']['logfile'])

    dimensions = get_dimensions(cfgstr['input']['datafile']) 
    dim_names = list(dimensions.keys())

    # Define chunk sizes based on the dimensions
    if len(dim_names) == 3:
        chunks = {dim_names[0]: dimensions[dim_names[0]] // 3, dim_names[1]: dimensions[dim_names[1]] // 2, dim_names[2]: dimensions[dim_names[2]] // 2}
    elif len(dim_names) == 2:
        chunks = {dim_names[0]: dimensions[dim_names[0]] // 2, dim_names[1]: dimensions[dim_names[1]] // 2}

    
    # Load either single or multiple dataset
    if cfgstr['input'].get('read_multiple_files', 'No') == 'Yes':
        ds = xr.open_mfdataset(cfgstr['input']['datafile'], chunks=chunks)
    else:
        ds = xr.open_dataset(cfgstr['input']['datafile'], chunks=chunks)


    # Remove 'uncertainty' if present in Dataset 
    if 'uncertainty' in ds.variables:
        print("Dropping 'uncertainty' variable from Dataset")
        ds = ds.drop_vars(['uncertainty'])
    else:
        print("'uncertainty' variable not found in Dataset. Skipping removal.")

    mylog.info('\n')
    # Print the size of the dataset 
    print('Original dataset size: ', ds.nbytes / (1024*1024*1024), 'GB')
    mylog.info("Original Dataset size: %s GB", ds.nbytes / (1024*1024*1024))

    # Calculate anomaly 
    try:
        if 'input' in cfgstr and cfgstr['input'] is not None:
            if 'cropped' in cfgstr['input']:
                if cfgstr['input']['cropped'] == 'Yes':
                    with ProgressBar():
                        cropped, anom_period_1, anom_period_2 = calculate_anomaly(ds,cfgstr,mylog)
                    
                    # Write to file
                    mylog.info('Writing anomaly periods to individual NetCDFs...')
                    write_to_netcdf(anom_period_1, cfgstr, 1)
                    write_to_netcdf(anom_period_2, cfgstr, 2)
                    mylog.info('NetCDFs have been written to output dir.')
                    

                elif cfgstr['input']['cropped'] == 'No':
                    mylog.info('Calculating anomaly periods...')
                    with ProgressBar():
                        anom_period_1, anom_period_2 = calculate_anomaly(ds,cfgstr,mylog)
                    mylog.info('Calculation of anomaly periods complete.')

                    mylog.info('Writing anomaly periods to individual NetCDFs...')
                    write_to_netcdf(anom_period_1, cfgstr, 1)
                    write_to_netcdf(anom_period_2, cfgstr, 2)
                    mylog.info('NetCDFs have been written to output dir.')


                elif cfgstr['input']['cropped'] == '':
                    print('cropped is empty in configuration file. Provide Yes/No.')
                    pass

                else:
                    print('cropped is not defined in the configuration file')
                    pass 
        else:
            print('input key in configuration file is None.')
            pass
            

    except Exception as e:
        mylog.error('Failed to calculate anomaly %s', e)
        raise
    

if __name__ == '__main__':
    main()
