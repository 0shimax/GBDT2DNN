import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


class GbdtDnn(nn.Module):
    def __init__(self, in_size, n_class=5, n_embeddings=2**13, n_h_embeddings=195):  # n_embeddings=2**(self.max_depth+1), 2**14
        super().__init__()

        self.in_size = in_size
        self.n_class = n_class
        self.n_h_embeddings = n_h_embeddings

        self.embedding = nn.Embedding(n_embeddings, n_h_embeddings)

        self.linear1 = nn.Linear(n_h_embeddings, 127)
        self.linear2 = nn.Linear(127, 83)
        self.linear3 = nn.Linear(83, 1)
        self.linear3_1 = nn.Linear(in_size, n_h_embeddings)
        self.linear4 = nn.Linear(n_h_embeddings, n_class)

        self.drop = nn.Dropout(0.5)

    def __call__(self, x):
        n_batch, _, _, n_features = x.shape
        h = self.embedding(x)
        h = h.reshape([-1, self.n_h_embeddings])
        h = F.relu(self.linear1(h))
        h = F.relu(self.linear2(h))
        h = F.relu(self.linear3(h))
        h = h.reshape([n_batch, n_features])
        h = F.relu(self.linear3_1(h))
        h = self.drop(h)
        h = self.linear4(h)
        return h
