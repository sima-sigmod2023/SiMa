from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from collections import OrderedDict
import pandas as pd
import os
from csv import DictReader
import json
import fasttext
import numpy as np
import torch
import random
import dgl
import torch.nn.functional as F
from multiprocessing import Pool
import itertools
import torch.nn as nn
import ast



def get_features(data: pd.DataFrame) -> pd.DataFrame:
    """
        Code for profiling tabular datasets and based on the profiler of Sherlock.

        Input:
            data: A pandas DataFrame with each row a list of string values
        Output:
            a dataframe where each row represents a column and columns represent the features
            computed for the corresponding column.
    """

    # Transform data so that each column becomes a row with its corresponding values as a list

    data = data.T
    list_values = data.values.tolist()
    data = pd.DataFrame(data={'values': list_values})

    data_columns = data['values']

    features_list = []

    for column in data_columns:

        column = pd.Series(column).astype(str)

        f = OrderedDict(list(extract_bag_of_characters_features(column).items()) + list(extract_bag_of_words_features(column, len(column)).items()))

        features_list.append(f)

    return pd.DataFrame(features_list).reset_index(drop=True)*1

def create_feature_tensor(col_features: dict(), col_ids: dict()):
    """

    :param col_features: Correspondences between columns and features
    :param col_ids: Correspondences between columns and ids
    :return: Correspondence between col ids and features in the form of a tensor
    """

    features_per_column = [[]] * len(col_ids)

    for col, features in col_features.items():
        features_per_column[col_ids[col]] = features

    return torch.tensor(features_per_column, dtype=torch.float)


def get_fasttext_embeddings(values, model_file):
    """
        Compute pre-trained embeddings for a list of values.
    """
    model = fasttext.load_model(model_file)

    f = OrderedDict()
    embeddings = []

    values = values.dropna()

    for v in values:

        v = str(v).lower()

        vl = v.split(' ')

        if len(vl) == 1:
            embeddings.append(model.get_word_vector(v))
        else:
            embeddings_to_all_words = []

            for w in vl:
                embeddings_to_all_words.append(model.get_word_vector(w))

            mean_of_word_embeddings = np.nanmean(embeddings_to_all_words, axis=0)
            embeddings.append(mean_of_word_embeddings)

    mean_embeddings = np.nanmean(embeddings, axis=0)

    for i, e in enumerate(mean_embeddings): f['word_embedding_avg_{}'.format(i)] = e

    return f


def get_embeddings(data: pd.DataFrame, model_file) -> pd.DataFrame:
    """
        Compute pre-trained FastText embeddings for each column of a tabular dataset,
        represented as a pandas dataframe.
    """
    data = data.T
    list_values = data.values.tolist()
    data = pd.DataFrame(data={'values': list_values})

    data_columns = data['values']

    embeddings_list = []

    for column in data_columns:
        column = pd.Series(column).astype(str)

        f = OrderedDict(list(get_fasttext_embeddings(column, model_file).items()))

        embeddings_list.append(f)

    return pd.DataFrame(embeddings_list).reset_index(drop=True) * 1


def generate_ids_paths(dir_path: str) -> (dict(), dict()):

    """
        Connects each column in the dataset to an id, and each file to each full_path

        Input:
            dir_path: The directory path under which we search for .csv datasets
        Output:
            columns_to_ids: A matching between columns and ids
            files_to_paths: A matching between filenames and full paths
    """

    columns_to_ids = dict()
    files_to_paths = dict()
    column_id = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".csv"):
                files_to_paths[file[:-4]] = os.path.join(root, file)
                with open(os.path.join(root, file), 'r') as read_obj:
                    csv_dict_reader = DictReader(read_obj)
                    column_names = csv_dict_reader.fieldnames

                    for c in column_names:
                        columns_to_ids[(file[:-4], c)] = column_id
                        column_id += 1

    return columns_to_ids, files_to_paths


def get_results(results_file: str, method: str):

    """
        Returns a list containing the matching results of the matching method to which the
        input .json file belongs.
    """

    def _parse_tuple(string):  # simple function to parse tuples
        try:
            s = ast.literal_eval(str(string))
            if type(s) == tuple:
                return s
            return
        except:
            return

    with open(results_file) as f:
        results_dict = json.load(f)

    results = []

    if method == 'EmbDI' or method == 'JaccardLevenMatcher':
        for k, v in results_dict.items():
            kt = _parse_tuple(k)
            results.append((kt[0], kt[1], v))
    else:
        for d in results_dict['results']:
            results.append(((d['source']['tbl_nm'], d['source']['clm_nm']),
                            (d['target']['tbl_nm'], d['target']['clm_nm']), d['sim']))

    # Normalize similarity scores
    amin, amax = results[-1][2], results[0][2]
    for i, val in enumerate(results):
        results[i] = (val[0], val[1], max(0, (val[2] - amin) / (amax - amin)))

    return results

def get_ground_truth(ground_truth_file: str):

    """
        Returns a set of matches that should hold among datasets of different categories.
    """


def columns_to_features(files_to_paths: dict()) -> dict():

    """
        Input:
            files_to_paths: Correspondences between files and their full paths
        Output:
            col_features: dictionary with column - features correspondence
    """

    col_features = dict()
    count = 1
    cols_count = len(files_to_paths)
    for file, filepath in files_to_paths.items():

        print('File ' + str(count) + "/" + str(cols_count))

        count += 1

        file_pd = pd.read_csv(filepath)

        cols = file_pd.columns.tolist()

        col_features_pd = get_features(file_pd)

        feature_list = col_features_pd.values.tolist()

        for i in range(len(cols)):

            col_features[(file, cols[i])] = feature_list[i]

    return col_features





def create_graphs(category_tables, cols_to_ids, graph_num, feat_list, ground_truth, no_datasets):
    """
        Function to create data silo configurations (and their corresponding relatedness graphs).

        Input:
            base_to_synthetic: dictionary which stores source_table - fabricated datasets correspondences
            cols_to_ids: dictionary with columns to ids correspondences
            graph_num: number of relatedness graphs (silos) to create
            feat_list: list containing column ids to features correspondences
            ground_truth: contains matches that should hold among columns of datasets belonging to different silos
            no_datasets: number of datasets to include per domain (source table)
        Output:
            graphs: relatedness graphs
            columns: columns included in each relatedness graph
            all_cols_ids: columns to ids correspondence for each relatedness graph
            all_ids_cols: inverted all_cols_ids
    """


    # dictionary holding the datasets that each relatedness graph includes
    samples = {i: [] for i in range(graph_num)}



    # sample relationships for each category
    for k, v in category_tables.items():
        l = len(v)

        if l // graph_num >= no_datasets:
            step = 0
            for _, s in samples.items():
                s.extend(random.sample(v[(step * l // graph_num):((step + 1) * l // graph_num)], no_datasets))
                step += 1
        else: # if there are not enough fabricated datasets for each category, some relatedness graphs receive datasets from more categories than others
            gn = graph_num
            while l // gn < no_datasets:
                gn = gn - 1
            step = 0
            for i, s in samples.items():
                if i >= (graph_num - gn):
                    print('Graph {} receives datasets from source {}'.format(i, k))
                    s.extend(random.sample(v[(step * l // gn):((step + 1) * l // gn)], no_datasets))
                    step += 1


    columns = {i: [] for i in range(graph_num)}

    for k, _ in cols_to_ids.items():

        for i, s in samples.items():
            if k[0] in s:
                columns[i].append(k)
                break

    all_cols_ids = dict()
    all_ids_cols = dict()

    for i, col in columns.items():
        count = 0
        d = dict()
        for c in col:
            d[c] = count
            count += 1
        all_cols_ids[i] = d
        invd = {v: k for k, v in d.items()}
        all_ids_cols[i] = invd

    features = {i: [[]] * len(columns[i]) for i in range(graph_num)}

    for i in range(graph_num):
        for c in columns[i]:
            features[i][all_cols_ids[i][c]] = feat_list[cols_to_ids[c]]


    edges = {i: [] for i in range(graph_num)}


    for i, cols in columns.items():

        for j in range(len(cols)):
            matched = False
            for k in range(j + 1, len(cols)):
                if cols[j][1] == cols[k][1] or (cols[j][1], cols[k][1]) in ground_truth or (
                cols[k][1], cols[j][1]) in ground_truth:
                    matched = True
                    edge1 = (all_cols_ids[i][cols[j]], all_cols_ids[i][cols[k]])
                    edge2 = (edge1[1], edge1[0])
                    edges[i].append(edge1)
                    edges[i].append(edge2)
            if not matched:
                edges[i].append((all_cols_ids[i][cols[j]], all_cols_ids[i][cols[j]]))


    graphs = dict()

    for i in range(graph_num):
        et = torch.tensor(edges[i], dtype=torch.long).t().contiguous()
        ft = torch.tensor(features[i], dtype=torch.float)
        graph = dgl.graph((et[0], et[1]))
        graph.ndata['feat'] = F.normalize(ft, 2, 0) # normalize input features
        graphs[i] = graph

    return graphs, columns, all_cols_ids, all_ids_cols

def metrics(count_tp, count_fp, count_fn):
    precision = (count_tp*1.0)/(count_tp + count_fp)
    recall = (count_tp*1.0)/(count_tp + count_fn)
    f_score = 2*precision*recall/(precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F-score: ' + str(f_score))



############ Functions for multiprocessing of GNN results############

def prediction_score(h1, h2, c11, c22, cid1, cid2, pred):
    first = h1[cid1[c11]]

    second = h2[cid2[c22]]
    hh = torch.cat([first, second])
    score = torch.sigmoid(pred.W2(F.relu(pred.W1(hh)))).detach().item()
    return score

def get_process_pairs(columns1, columns2, h1, h2, cid1, cid2, pred):
    for c11, c22 in itertools.product(columns1, columns2):
        yield c11, c22, h1, h2, cid1, cid2, pred


def process_score(input_tuple):
    c11, c22, h1, h2, cid1, cid2, pred = input_tuple
    score = prediction_score(h1, h2, c11, c22, cid1, cid2, pred)
    return c11, c22, score


def run_multithread(columns1, columns2, h1, h2, cid1, cid2, pred, no_threads):
    with Pool(no_threads) as process_pool:
        similarities = process_pool.map(process_score, get_process_pairs(columns1, columns2, h1, h2, cid1, cid2, pred))

    return similarities

############ Functions for multiprocessing of baseline results############

def get_baseline_pairs(columns1, columns2, h1, h2, cid1, cid2):
    for c11, c22 in itertools.product(columns1, columns2):
        yield c11, c22, h1, h2, cid1, cid2


def baseline_score(h1, h2, c11, c22, cid1, cid2):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    first = h1[cid1[c11]]

    second = h2[cid2[c22]]

    return cos(first, second).detach().item()


def baseline_process_score(input_tuple):
    c11, c22, h1, h2, cid1, cid2 = input_tuple
    score = baseline_score(h1, h2, c11, c22, cid1, cid2)
    return c11, c22, score


def run_baseline_multithread(columns1, columns2, h1, h2, cid1, cid2, no_threads):
    with Pool(no_threads) as process_pool:
        similarities = process_pool.map(baseline_process_score,
                                        get_baseline_pairs(columns1, columns2, h1, h2, cid1, cid2))

    return similarities











