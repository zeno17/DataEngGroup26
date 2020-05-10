import sys
from pyspark import SparkConf, SparkContext
import scipy
import numpy as np
from math import sqrt
import time

""""
Helper methods:
"""
MIN_DATA_POINTS = 28
def prep_file(files):
    """"Method to preprocess the file based on the stock dataset
    parameters:
        file_name - file name
    Return ([[date_time],[value]],has_enough_lines)
        date_time - date and time of stock
        value - closing prices at that time
        has_enough_lines - boolean specifying if there are more than 10 datapoints"""

    #Split each file into single lines (datapoints)
    lines = []
    for file in files:
        lines.append(file.split("\r\n"))


    date_times = []
    vals = []
    counter = 0
    avg_cnt = 0
    avg_val = 0
    for file in lines:
        prev_time = ""
        file = [line for line in file if line]
        prev_date = file[0].split(",")[0]
        for line in file:
            split = line.split(",")
            duplicate = prev_date == split[0]
            if not duplicate:
                date_times.append(prev_date)
                vals.append(avg_val / avg_cnt)
                avg_val = 0
                avg_cnt = 0
                counter += 1
            avg_cnt += 1
            avg_val += float(split[5])
            prev_date = split[0]
        date_times.append(prev_date)
        vals.append(avg_val / avg_cnt)
    date_times, idx = np.unique(date_times, return_index = True)
    values = np.take(vals, idx)
    if counter > MIN_DATA_POINTS and len(values) == len(date_times):
        return ([date_times, values], True)
    else:
        return ([date_times, values], False)

def post_prep(closing_prices):
    empty_dates = []
    for cp in range(len(closing_prices)):
        if closing_prices[cp][1] == 0:
            empty_dates.append(cp + 1)

    first = 0
    for x in range(len(closing_prices)):
        if (x+1) not in empty_dates:
            first = x
            break

    for x in range(first):
        closing_prices[x][1] = closing_prices[first][1]

    last_price = 0
    new_closing_prices = []
    # CHANGED THIS FOR-LOOP
    for x in range(len(closing_prices)):
        if closing_prices[x][1] == 0:
            new_closing_prices.append(float(last_price))
        else:
            last_price = closing_prices[x][1]
            new_closing_prices.append(float(last_price))
    return new_closing_prices

def find_month(date):
    jan_dates = create_dates(1,31) #Create dates for january
    feb_dates = create_dates(2, 29) #Create dates for february
    mar_dates = create_dates(3, 31) #Create dates for march
    apr_dates = create_dates(4, 30) #Create dates for april
    may_dates = create_dates(5, 31) #Create dates for may

    if date in jan_dates:
        cur_month = jan_dates
    elif date in feb_dates:
        cur_month = feb_dates
    elif date in mar_dates:
        cur_month = mar_dates
    elif date in apr_dates:
        cur_month = apr_dates
    elif date in may_dates:
        cur_month = may_dates
    else:
        return AssertionError
    return cur_month

def create_dates(month, days):
    if month < 10:
        x = "0%d" % month
    else:
        x = str(month)
    dates = ["%s/0%d/2020" % (x, i) for i in range(1, 10)]

    for x in ["%s/%d/2020" % (x, i) for i in range(10, days+1)]:
        dates.append(x)
    return dates

def find_first_date_number(first_date, cur_month):
    counter = 0
    for date in cur_month:
        if first_date is date:
            return counter
        else:
            counter += 1
    return 31


def read_file(file_name):
    file = open("%s/%s.txt" % (STOCK_DIR, file_name))
    lines = file.readlines()
    file.close()
    print(lines)
    return lines


def load_stock_file(filename):
    file = sc.textFile("%s/filename" % (STOC_DIR))
    file_par = filename.parallelize(file.collect())
    return file_par

def get_indices(ind, matrix):
    listed = []
    if ind != 0:
        for i in range(ind):
            listed.append(matrix[i][ind])

    listed = listed + matrix[ind][ind+1:]# add +1 to remove diagonal
    return listed

def transform_tuple(tup):
    key = tup[0]
    indices_tup = tup[1][1]
    stocks = tup[1][0]
    tuple_to_ret = ((key, stocks), indices_tup)
    return tuple_to_ret

def swap_key_values(tup):
    key = tup[0]
    value = tup[1]
    return (value, key)

def init_table_indices(n):
    array = []
    for i in range(n):
        array.append(list(range(i*n, (i+1)*n)))
    return array

def get_average(dataset):
    average = 0
    for x in range(len(dataset)):
        average += dataset[x]
    return average / len(dataset)

def prep_for_corr_pearson(input1, input2):
    '''
    input1 = [[<dates], [<values>]]
    input2 = [[<dates], [<values>]]
    Computes data based on values for which the date is present in both inputs
    '''
    if input1 is None or input2 is None:
        return 100
    matches, ind_inp1, ind_inp2 = np.intersect1d(input1[0], input2[0], return_indices=True)
    series1 = np.array(input1[1])[ind_inp1]
    series2 = np.array(input2[1])[ind_inp2]
    return pearsonrcustom(series1, series2)

def prep_for_corr_spearman(input1, input2):
    '''
    input1 = [[<dates], [<values>]]
    input2 = [[<dates], [<values>]]
    Computes data based on values for which the date is present in both inputs
    '''
    if input1 is None or input2 is None:
        return 100
    matches, ind_inp1, ind_inp2 = np.intersect1d(input1[0], input2[0], return_indices=True)
    series1 = np.array(input1[1])[ind_inp1]
    series2 = np.array(input2[1])[ind_inp2]
    return cust_spearmanr(series1, series2)


def pearsonrcustom(x, y):
    # Custom function based on scipy framework
    if len(x) < 1 or len(y) < 1:
        return 101
    n = len(x)
    x = np.asarray(x)
    y = np.asarray(y)

    dtype = type(1.0 + x[0] + y[0])
    xmean = np.mean(x)
    ymean = np.mean(y)

    xm = x - xmean
    ym = y - ymean

    normxm = np.linalg.norm(xm)
    normym = np.linalg.norm(ym)
    r = np.dot(xm / normxm, ym / normym)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)
    return r


def cust_spearmanr(x, y):
    if len(x) < 1 or len(y) < 1:
        return 101
    ranksx = ranking(x)
    ranksy = ranking(y)
    r = pearsonrcustom(ranksx, ranksy)
    return r


def ranking(x):
    arr = np.ravel(np.asarray(x))
    sorter = np.argsort(arr, kind='mergesort')

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)

start = time.time()

""""
    Set the paramters and variables for spark structure:
"""
# Spark context settings
appName = "Data Engineering"
master = 'local'

conf = SparkConf().setAppName(appName).setMaster(master)
conf.set("spark.executor.memory", "5g")
conf.set("spark.driver.memory", "5g")
conf.set("spark.driver.maxResultSize", "6g")
conf.set("spark.network.timeout", "3600s")
conf.set("spark.executor.heartbeatInterval","3599s")
sc = SparkContext('local[4]',conf=conf)

# Set the directory paths
directory_path_2020 = "C:/Users/tzvet/OneDrive - TU Eindhoven/Masters Year 1/Q4/2IMD15/Milestones/Milestone_1/2020/"
directory_path_2018 = "C:/Users/tzvet/OneDrive - TU Eindhoven/Masters Year 1/Q4/2IMD15/Milestones/Milestone_1/2018/"
directory_path_2020_test = "C:/Users/tzvet/OneDrive - TU Eindhoven/Masters Year 1/Q4/2IMD15/Milestones/Milestone_1/2020_test/"
directory_path_2020_1KB = "C:/Users/tzvet/OneDrive - TU Eindhoven/Masters Year 1/Q4/2IMD15/Milestones/Milestone_1/2020_1KB/"
directory_path_2020_no_probl = "C:/Users/tzvet/OneDrive - TU Eindhoven/Masters Year 1/Q4/2IMD15/Milestones/Milestone_1/2020_no_probl/"

# Final path chosen
final_path = directory_path_2020 # change this to set the directory you want

# Get the file names of all the files in the given directory
path = final_path
fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
list_status = fs.listStatus(sc._jvm.org.apache.hadoop.fs.Path(path))
file_names = [file.getPath().getName() for file in list_status]
stock_names = []
set_stock_names = set(stock_names)

for file_name in file_names:
    name = ('_'.join(file_name.split('_')[1:])).split('.')[0].lower()
    if name not in set_stock_names:
        set_stock_names.add(name)
        stock_names.append(name)

num_files = len(stock_names)
#print(stock_names)

# Initilialize 2D array containing the indices to be set
table_indices = init_table_indices(num_files)

list_tuple_table_indices = []
for file in stock_names:
    index_file = stock_names.index(file)
    indices = get_indices(index_file, table_indices)
    list_tuple_table_indices.append((file, indices))

# Threshold
T = 0.95

start_spark = time.time()
""""
    Spark structure:
"""

# Convert list_tuole_indices to RDD to join in the next step
rdd_tuple_indices = sc.parallelize(list_tuple_table_indices)


final = sc.wholeTextFiles(final_path)\
    .map(lambda pair: ('_'.join(pair[0].split('/')[-1].split('_')[1:]).split('.')[0].lower(), pair[1]))\
    .groupByKey().map(lambda pair : (pair[0], list(pair[1]))).map(lambda pair: (pair[0], prep_file(pair[1])))\
    .filter(lambda pair: pair[1][1] == True).map(lambda pair: (pair[0], pair[1][0]))\
    .rightOuterJoin(rdd_tuple_indices).map(lambda pair: transform_tuple(pair)).flatMapValues(lambda x: x)\
    .map(lambda pair: swap_key_values(pair))

    #.reduceByKey(lambda accum, st: accum + st)\ # This goes instead of reduceByKey
    #.map(lambda pair: (pair[0], prep_file(pair[1])))\
    #.filter(lambda pair: pair[1][1] == True).map(lambda pair: (pair[0], pair[1][0]))\
    #.rightOuterJoin(rdd_tuple_indices).map(lambda pair: transform_tuple(pair)).flatMapValues(lambda x: x) \
    #.map(lambda pair: swap_key_values(pair))#.reduceByKey(lambda tup1, tup2: pearson_correlation(tup1[1], tup2[1]))

rdd_tuple_indices.unpersist()
print("Preprocessing takes: " + str(time.time() - start_spark))


#final.saveAsTextFile(final_path+"Final")
print("Pearson")
start_pearson = time.time()
pearson = final.reduceByKey(lambda tup1, tup2: prep_for_corr_pearson(tup1[1], tup2[1]))\
    .filter(lambda pair: pair[1] < 2 and pair[1] > -2).filter(lambda pair: pair[1] >= T or pair[1] <= T)\
    .sortBy(lambda pair: pair[1], ascending=False).take(10)
print("COmpiled pearson in: " + str(time.time()-start_pearson))


print("Spearman")
start_spearman = time.time()
spearman = final.reduceByKey(lambda tup1, tup2: prep_for_corr_spearman(tup1[1], tup2[1]))\
    .filter(lambda pair: pair[1] < 2 and pair[1] > -2).filter(lambda pair: pair[1] >= T or pair[1] <= T)\
    .sortBy(lambda pair: pair[1], ascending=False).take(10)
print("COmpiled spearman in: " + str(time.time()-start_spearman))

final.unpersist()
sys_time = time.time()
#print("Pearson")
#print(pearson)
#pearson.saveAsTextFile(final_path+"Pearson_"+str(sys_time))
#pearson.unpersist()

#print("Spearman")
#print(spearman)
#spearman.saveAsTextFile(final_path+"Spearman_"+str(sys_time))
#spearman.unpersist()
sc.stop()
print("Total time: " + str(time.time()-start_spark))


def get_stock_names(cell_index):
    for row in range(num_files):
        for col in range(num_files):
            if table_indices[row][col] == cell_index:
                return stock_names[row], stock_names[col]

pearson_top_stocks = []
for res in pearson:
    stock_1, stock_2 = get_stock_names(res[0])
    pearson_top_stocks.append(((stock_1, stock_2), res[1]))

spearman_top_stocks = []
for res in spearman:
    stock_1, stock_2 = get_stock_names(res[0])
    spearman_top_stocks.append(((stock_1, stock_2), res[1]))

print("TOP PEARSON STOCKS")
print(pearson_top_stocks)

print("TOP SPEARMAN STOCKS")
print(spearman_top_stocks)

with open("Top pearson.txt", 'w') as filehandle_pearson:
    filehandle_pearson.writelines("%s\n" % str(res) for res in pearson_top_stocks)

with open("Top spearman.txt", 'w') as filehandle_spearman:
    filehandle_spearman.writelines("%s\n" % str(res) for res in spearman_top_stocks)