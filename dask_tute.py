import os, psutil, seaborn as sns, numpy as np, pandas as pd, matplotlib.pyplot as plt, glob, re

#Writing a memory output function
def mem_stats():
    mem = psutil.Process(os.getpid()).memory_info().rss
    return (mem / 1024**2)

#Load a dummy dataset
df = sns.load_dataset('tips')

#Evaluate the size of an array
print(np.array(df.tip).nbytes / 1024**2)

#Check mem footprint before operation
before = mem_stats()

#Performing basic calc
tips2 = np.array(df.tip) / 1.2 + 2

#After operation
after = mem_stats()

#Calc on additional mem overhead
after - before

#Example chunking
taxi_df_path = 'yellow_tripdata_2015_agg.csv'
for chunk in pd.read_csv(taxi_df_path, chunksize = 50000):
    print('type: %s shape %s' % (type(chunk), chunk.shape))

#Defining a filtering criteria
def filter_long_trip(data):
    long_trip = (data.trip_distance > 3)
    return data.loc[long_trip]

#Running a list comp to filter chunks for a given criteria
chunks = []
chunks = [filter_long_trip(chunk) for chunk in pd.read_csv(taxi_df_path, chunksize = 50000)]
chunk_filtered_df = pd.concat(chunks)

#Plotting trip times vs distance
chunk_filtered_df['dropoff'] = pd.to_datetime(chunk_filtered_df['tpep_dropoff_datetime'])
chunk_filtered_df['pickup'] = pd.to_datetime(chunk_filtered_df['tpep_pickup_datetime'])
#Creating a trip time
chunk_filtered_df['trip_time_min'] = chunk_filtered_df['dropoff'] - chunk_filtered_df['pickup']
#Converting to seconds
chunk_filtered_df['trip_time_sec'] = chunk_filtered_df['trip_time_min'] / np.timedelta64(1, 's')
#Correcting for erroneous results
chunk_filtered_df = chunk_filtered_df.loc[(chunk_filtered_df.trip_distance < 600000) & (chunk_filtered_df.trip_time_sec > 0)]
#Final plot
sns.scatterplot(x = 'trip_distance', y = 'trip_time_sec', data = chunk_filtered_df)

'''Generators - these use lazy evaluation, elements are generated one at a time
this means they are never simultaneously in memory. Generators are consumed once
operations are performed on them'''

#Alternate workflow incorporating Generators
template = 'nyctaxi/yellow_tripdata_2015-{:02d}.csv'
#generator of files in the location
files = (template.format(k) for k in range(1, 4))
for fname in files:
    print(fname) 

def long_trips(df):
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.seconds
    long_duration = df['duration'] > 1200
    result_dict = {'n_long': [sum(long_duration)],
                    'n_total': [len(df)]}
    return pd.DataFrame(result_dict)

#reuse file generator
files = (template.format(k) for k in range(1, 4))
dfs = (pd.read_csv(fname, parse_dates = [1, 2])
            for fname in files)
totals = (long_trips(df) for df in dfs)
#What proportion of trips were longer than 20 mins?
q1_totals = sum(totals)
q1_totals['n_long'] / q1_totals['n_total']

#Simpler with flights data
def prp_delay(df):
    n_del = (df['DEP_DELAY'] > 0).sum()
    return n_del * 100 / len(df)

template_2 = 'flightdelays/flightdelays-2016-{:1d}.csv'
files_2 = (template_2.format(k) for k in range(1, 6))
dataframes = (pd.read_csv(fname, parse_dates = [0]) for fname in files_2)
md = [prp_delay(df) for df in dataframes]

#Plotting the proportion of delayed flights
x = range(1, 6)
plt.plot(x, md, marker = 'o', linewidth = 0)
plt.ylabel('% delayed')
plt.xlabel('Month - 2016')
plt.ylim(0, 100)
plt.xlim(1, 5)
plt.show()

#Pipelining the same using delayed functions for dask
from dask import delayed

@delayed
def count_flights(df):
    return len(df)

@delayed
def count_delayed(df):
    return (df['DEP_DELAY'] > 0).sum()

@delayed
def pct_delayed(n_delayed, n_flights):
    return 100 * sum(n_delayed) / sum(n_flights)

#creating a reading function
@delayed
def read_file(fname):
    return pd.read_csv(fname)

#Reusing the files 2 generator
files_2 = (template_2.format(k) for k in range(1, 6))

#Building the dask pipeline
n_delayed = []
n_flights = []

for file in files_2:
    df = read_file(file)
    n_delayed.append(count_delayed(df))
    n_flights.append(count_flights(df))

result = pct_delayed(n_delayed, n_flights)

#No computation done until the invocation of .compute()
print(result.compute())

#Working with dask arrays
import h5py, dask.array as da, time

#Preparing hdf5 for reading
f = h5py.File('Texas/texas.2000.hdf5', 'r')

#Checking available datafiles
for key in f.keys():
    print(key)

#Loading in the 'load' file data
data = f['load'][:]

#closing the hdf5
f.close()

#Creating dask arrays, chunks of 7
nrg_2k = da.from_array(data, chunks = data.shape[0] // 7)

#Timing mean computation across the 7 dask arrays
t_st = time.time(); \
mean7 = nrg_2k.mean().compute(); \
t_ed = time.time()
telp = (t_ed - t_st) * 1000
print('time elapsed: {} ms'.format(round(telp, 3)))

#Size of dask_array and lengths of chunks
len(nrg_2k)
nrg_2k.chunks

#Reading in all load files at once
files = sorted(glob.glob('Texas/*.hdf5'))
dsets = [h5py.File(f)['/load'] for f in files]
#Reshaping as a numpy array to give yearly, daily and 15 minute intervals of power readings
arrs = [np.array(d).reshape((1, 365, 96)) for d in dsets[1:]]
#Stacking into 4 years
arrs_stacked = np.concatenate(arrs, axis = 0)
#converting to yearly array, with first dimension as year
da_arrs = da.from_array(arrs_stacked)

#Alternative workflow all in dask with no intermediate NumPy
da_arrs2 = [da.from_array(d) for d in dsets[1:]]
da_arrs2_stack = da.stack(da_arrs2)
da_arrs2_rshp = da.reshape(da_arrs2_stack, (4, 365, 96))

#Working with dask dataframes
import dask.dataframe as dd

df = dd.read_csv('WDI.csv')

#Looking at all indicator filters
np.array(df['Indicator Code'].unique())
fil1 = df['Indicator Code'] == 'SP.POP.0014.TO.ZS'
fil2 = df['Region'] == 'East Asia & Pacific'

#Filtering
df1 = df.loc[fil1 & fil2]

#Basic grouping and plotting output
gp1 = df1.groupby('Year').mean()
gp2 = gp1.compute()
gp2['value'].plot.line()
plt.ylim(0, 100)
plt.show()

#Big data storage
#>10TB requires RAID array or clustered computing
#Timing exercises
def yng_region(df):
    fil1 = df['Indicator Code'] == 'SP.POP.0014.TO.ZS'
    fil2 = df['Year'] ==  2015
    regions = df.loc[fil1 & fil2].groupby('Region')
    return regions['value'].mean()

t0 = time.time()
df = pd.read_csv('WDI.csv')
result = yng_region(df)
t1 = time.time()
print((t1 - t0) * 1000)

t0 = time.time()
df = dd.read_csv('WDI.csv')
result = yng_region(df)
t1 = time.time()
print((t1 - t0) * 1000)

#Reading in the NYC data
df = dd.read_csv('nyctaxi/*.csv', assume_missing = True)
df['tip_frac'] = df['tip_amount'] / (df['total_amount'] - df['tip_amount'])
df['tpep_pickup_datetime'] = dd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = dd.to_datetime(df['tpep_dropoff_datetime'])
df['hour'] = df['tpep_dropoff_datetime'].dt.hour

#Finding tips by hour
df1 = df.loc[df['payment_type'] == 1]
rs1 = df1.groupby('hour')['tip_frac'].mean()
rs2 = rs1.compute()

rs2.plot.line()
plt.ylim(0.1, 0.5)
plt.show()

#Working with dask bags
import dask.bag as db

files = sorted(glob.glob('sotu/*.txt'))
speeches = db.read_text(files)

#Basic operations with bags
speeches.count().compute()

#Taking elements
el1 = speeches.take(1)
el2 = el1[0]
el2[:60]

#Further ops over all the bags
word = speeches.str.split(' ')
n_words = word.map(len)
n_words.mean().compute()

#Filtering by speech content
spch_lwr = speeches.str.lower()
health = spch_lwr.filter(lambda s:'health care' in s)
health.count().compute()

#Working with JSON
import json

#Reading in as dask bags
cong_text = db.read_text('congress/bills*.json')

#Converting to JSON
cong_dicts = cong_text.map(json.loads)

#Extracting elements and reading keys
ex1 = cong_dicts.take(1)[0]
ex1.keys()

#Calculating average bill length
def bill_length(d):
    curr = pd.to_datetime(d['current_status_date'])
    intr = pd.to_datetime(d['introduced_date'])
    return(curr - intr).days

days = cong_dicts.filter(lambda x: x['current_status'] == 'enacted_signed').map(bill_length)
days.mean().compute()
