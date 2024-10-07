import pandas as pd
import scipy
import numpy as np
import multiprocessing
from scipy.cluster.hierarchy import average,fcluster
from scipy.spatial.distance import cdist,squareform
import sys
import argparse


def worker_cluster(start, end, distance_matrix, result, total_rows):
    for i in range(start, end):
        ave = average(distance_matrix[i])
        cluster = fcluster(ave, 0, criterion='distance')
        result[i] = cluster
        if i % 100 == 0:
            print(f"Process {multiprocessing.current_process().name}: {i}/{total_rows} clusters calculated")


def cal_cluster_parallel(distance_matrix):
    n = distance_matrix.shape[0]
    cluster_result = multiprocessing.Array('i', n)
    cluster_result_np = np.frombuffer(cluster_result.get_obj(), dtype=int)

    num_cores = multiprocessing.cpu_count()
    num_processes = int(num_cores * 0.8)  # 使用 CPU 核心數的 80%
    chunk_size = n // num_processes

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else n
        p = multiprocessing.Process(target=worker_cluster,
                                    args=(start, end, distance_matrix, cluster_result_np, n))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return cluster_result_np

def worker(start, end, data, result, total_rows):
    for i in range(start, end):
        distances = np.sum(data[i] != data, axis=1)
        result[i, :] = distances
        result[:, i] = distances
        if i % 100 == 0:
            print(f"Process {multiprocessing.current_process().name}: {i}/{total_rows} rows processed")


def hamming_distance(row1, row2):
    return np.sum(np.bitwise_xor(row1,row2))

def cal_distance_matrix(data):
    distance_matrix = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        if i % 100 == 0:
            print("start calculate "+str(i)+" row")
        for j in range(i+1, data.shape[0]):
            distance = hamming_distance(data[i,:], data[j,:])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix
    
def cal_distance_matrix_parallel(data):
    n = data.shape[0]
    distance_matrix = multiprocessing.Array('d', n*n)
    distance_matrix = np.frombuffer(distance_matrix.get_obj()).reshape((n, n))

    num_cores = multiprocessing.cpu_count()
    num_processes = int(num_cores * 0.8)  # 使用 CPU 核心數的 80%

    chunk_size = n // num_processes

    processes = []
    for i in range(num_processes):
        start = i * chunk_size
        end = start + chunk_size if i < num_processes - 1 else n
        p = multiprocessing.Process(target=worker, args=(start, end, data, distance_matrix, n))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return distance_matrix

def convert_data_to_int(data):
    data_int = np.zeros(data.shape, dtype=int)
    for i in range(data.shape[1]):
        catg_map = dict()
        catg_index=0
        for j in range(data.shape[0]):
            if data[j][i] not in catg_map:
                catg_map[data[j][i]] = catg_index
                catg_index += 1
            data_int[j][i] = catg_map[data[j][i]]
    return data_int

def cal_cluster(distance_matrix):
    condensed_distance_matrix = squareform(distance_matrix)
    
    ave = average(condensed_distance_matrix)
    cluster = fcluster(ave, 0, criterion='distance')
    return cluster
    # ave = average(distance_matrix)
    # cluster = fcluster(ave, 0, criterion='distance')
    # return cluster


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_path", default='converted_values_efficient_cluster.csv', help="input csv path")
    parser.add_argument("-o","--output_path", default='converted_values_efficient_cluster.csv', help="output csv path")
    parser.add_argument("-d","--data_point", help="how many sample to calculate, leave it empty if you wish to calculate whole input file", default="")
    parser.add_argument("-f","--f", default='', help="for jupyter notebook")
    args = parser.parse_args()
    return args

def main():
    # args
    args = args_parse()
    print("="*30)
    print('data_point:',args.data_point)
    print('input_path:', args.input_path)
    print('output_path:', args.output_path)
    print("="*30)

    # load csv to dataframe
    df = pd.read_csv(args.input_path)
    if args.data_point != "":
        try:
            df = df.iloc[:int(args.data_point)]
        except Exception as e:
            print("data point should be empty or number")
    print(f"df.shape: {df.shape}")
    data = df.to_numpy()[:, 1:-1] # [:,1:]
    print(f"data.shape: {data.shape}")
    # avoid string/float comparison for faster hamming distance calculation
    data_int = convert_data_to_int(data)
    # distance_matrix = cal_distance_matrix(data_int)
    distance_matrix = cal_distance_matrix_parallel(data_int)

    print("distance_matrix finished!")

    # clustering
    cluster = cal_cluster(distance_matrix)
    #cluster = cal_cluster_parallel(distance_matrix)

    df['cluster'] = cluster
    print(df)
    df.to_csv(args.output_path)

if __name__ == '__main__':
    main()

