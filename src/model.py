import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import itertools

############ Define GNN model ############

class GraphSAGE(nn.Module):
    """
        GraphSAGE model used in Casanova:
            - 1 convolutional layer
            - Normalization applied in the output
    """

    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, h_feats, 'pool')


    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.normalize(h)
        return h

############ Define prediction model ############

class MLPPredictor(nn.Module):
    """
        Multi-layer Perceptron predictor used in Casanova:
            - 1 hidden layer
    """


    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


############ Define Casanova model ############

class CasanovaModel():
    """
        Casanova model consisting of:
            - GraphSAGE for computing embeddings of nodes
            - Predictor model for link prediction between nodes
            - Optimizer initialization (Adam)
    """
    def __init__(self, in_feats, h_feats, lr_rate, wd):
        self.gnn = GraphSAGE(in_feats, h_feats)
        self.predictor = MLPPredictor(h_feats)
        self.optimizer = torch.optim.Adam(itertools.chain(self.gnn.parameters(), self.predictor.parameters()), lr=lr_rate, weight_decay=wd)

