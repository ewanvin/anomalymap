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
    png_output = cfgstr['output']['png_output']

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

    if 'data_endpoint' in cfgstr['input'] and cfgstr['input']['data_endpoint']:
        print('Reading OpenDAP link...')
        link = cfgstr['input']['data_endpoint']
         
        # Slicing Dataset
        cropped = dataset.sel(xc=slice(0, -3250))
        
        print(cropped.dims)

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

        mylog.info('Calculating anoamlies for each period...')
        # Calculate the anomalies for each period
        anom_period_1 = period_1.mean(dim='time') - mean
        anom_period_1 = xr.where(np.isnan(anom_period_1), np.nan, anom_period_1)
        
        anom_period_2 = period_2.mean(dim='time') - mean
        anom_period_2 = xr.where(np.isnan(anom_period_2), np.nan, anom_period_2)
        mylog.info('Calculation of anoamlies completed.')
              
        return cropped,anom_period_1, anom_period_2 
    

    elif len(dataset.dims) > 2:
        print('The dataset has more than 4 dimensions')
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
     


def create_dataset(dataset, anom_period_1, anom_period_2):
    # Assign new variables
    dataset['anom_period_1'] = anom_period_1
    dataset['anom_period_2'] = anom_period_2

    return dataset

def create_cropped_dataset(cropped, anom_period_1, anom_period_2):
    # Assign new variables 
    cropped['anom_period_1'] = anom_period_1 
    cropped['anom_period_2'] = anom_period_2

    return cropped

def main():

    args = parse_arguments()
    cfgstr = parse_cfg(args.cfgfile)
    mylog = initialise_logger(cfgstr['output']['logfile'])

    # Load dataset 
    ds = xr.open_dataset(cfgstr['input']['datafile'], chunks={'xc': 1300, 'yc': 1300, 'time': 1000})
    #ds = xr.open_mfdataset(cfgstr['input']['datafile'], chunks={'time': 3, 'lat': 800, 'lon': 12000}) 

    # Remove 'uncertainty' if present in dataset
    if 'uncertainty' in ds.variables:
        print("Dropping 'uncertainty' variables from Dataset")
        ds = ds.drop_vars(['uncertainty'])
    else:
        print("'uncertainty' variable not found in Dataset. Skipping removal.")

        

    mylog.info('\n')
    # Print the size of the dataset 
    print('Original dataset size: ', ds.nbytes / (1024*1024*1024), 'GB')
    mylog.info("Original Dataset size: %s GB", ds.nbytes / (1024*1024*1024))

    # Calculate anomaly 
    try:
        if 'data_endpoint' in cfgstr['input'] and cfgstr['input']['data_endpoint']:
            with ProgressBar():
                cropped, anom_period_1, anom_period_2 = calculate_anomaly(ds,cfgstr,mylog)
            
            # Add anomaly periods to dataset
            mylog.info('Creating new cropped dataset...')
            cropped_new = create_cropped_dataset(cropped, anom_period_1, anom_period_2)
            mylog.info('Creation of new cropped dataset complete.')
            
            # Write to file
            mylog.info('Writing new dataset to NetCDF...')
            with ProgressBar(), ResourceProfiler(dt=0.25) as rprof:
                cropped_new.to_netcdf(cfgstr['output']['datafile'], encoding={'sca': {'zlib': True, 'complevel': 9, 'chunksizes': (500, 450, 700), 'dtype': 'float64'},
                    'time_bounds': {'zlib': True, 'complevel': 9, 'chunksizes': (500,2), 'dtype': 'float64'},
                    'lon': {'zlib': True, 'complevel': 9, 'chunksizes': (450, 700), 'dtype': 'float64'},
                    'lat': {'zlib': True, 'complevel': 9, 'chunksizes': (450, 700), 'dtype': 'float64'},
                    'lmask': {'zlib': True, 'complevel': 9, 'chunksizes': (450, 700), 'dtype': 'float64'},
                    'anom_period_1': {'zlib': True, 'complevel': 9, 'chunksizes': (450, 700), 'dtype': 'float64'},
                    'anom_period_2': {'zlib': True, 'complevel': 9, 'chunksizes': (450, 700), 'dtype': 'float64'}})
            mylog.info('NetCDF have been written to output dir.')
        else:
            mylog.info('Calculating anomaly periods...')
            with ProgressBar(), ResourceProfiler(dt=0.25) as rprof:
                anom_period_1, anom_period_2 = calculate_anomaly(ds,cfgstr,mylog)
            mylog.info('Calculation of anomaly periods complete.')

            

            mylog.info('Creating new dataset...')
            ds_new = create_dataset(ds, anom_period_1, anom_period_2)
            mylog.info('Creation of new dataset complete.')
                        
            mylog.info('Writing new dataset to NetCDF...')
            with ProgressBar(), ResourceProfiler(dt=0.25) as rprof:
                ds_new.to_netcdf(cfgstr['output']['datafile'], encoding={'FSC_STD': {'zlib': True, 'complevel': 9, 'chunksizes': (1, 350, 600), 'dtype': 'float64'},
                'Bit_Flags': {'zlib': True, 'complevel': 9, 'chunksizes': (1,350,600), 'dtype': 'float64'},
                'FSC': {'zlib': True, 'complevel': 9, 'chunksizes': (1,350,600), 'dtype': 'float64'}})
            mylog.info('NetCDF have been written to output dir.')

            rprof.visualize()

    except Exception as e:
        mylog.error('Failed to calculate anomaly %s', e)
        raise
    

if __name__ == '__main__':
    main()