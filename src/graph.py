import tools
import torch
import dgl
import random
import torch.nn.functional as F



def create_graphs(base_to_synthetic, cols_to_ids, graph_num, feat_list, emb_list, ground_truth, no_datasets,
                  partitioning, domain):
    """
        Function to create data silo configurations (and their corresponding relatedness graphs).

        Input:
            base_to_synthetic: dictionary which stores source_table - fabricated datasets correspondences
            cols_to_ids: dictionary with columns to ids correspondences

    """



    samples = {i: [] for i in range(graph_num)}

    if partitioning == 'equal-datasets' and domain == 'equal-mixed':

        # sample relationships for each category
        for k, v in base_to_synthetic.items():
            l = len(v)

            if l // graph_num >= no_datasets:
                step = 0
                for _, s in samples.items():
                    s.extend(random.sample(v[(step * l // graph_num):((step + 1) * l // graph_num)], no_datasets))
                    step += 1
            else:
                gn = graph_num
                while l // gn < no_datasets:
                    gn = gn - 1
                step = 0
                for i, s in samples.items():
                    if i >= (graph_num - gn):
                        print('Graph {} receives datasets from source {}'.format(i, k))
                        s.extend(random.sample(v[(step * l // gn):((step + 1) * l // gn)], no_datasets))
                        step += 1
    elif partitioning == 'equal-datasets' and domain == 'gradually-mixed':
        # sample relationships for each category
        graph = 0

        for k, v in base_to_synthetic.items():

            l = len(v)
            step = 0
            for i, s in samples.items():
                if graph <= i:
                    s.extend(random.sample(v[(step * l // graph_num):((step + 1) * l // graph_num)], no_datasets))
                step += 1
            graph += 1

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
    embeddings = {i: [[]] * len(columns[i]) for i in range(graph_num)}

    for i in range(graph_num):
        for c in columns[i]:
            features[i][all_cols_ids[i][c]] = feat_list[cols_to_ids[c]]
            embeddings[i][all_cols_ids[i][c]] = emb_list[cols_to_ids[c]]

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
        graph.ndata['embeddings'] = torch.tensor(embeddings[i], dtype=torch.float)
        graphs[i] = graph

    return graphs, columns, all_cols_ids, all_ids_cols


