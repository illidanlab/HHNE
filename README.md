# HHNE
Heterogeneous hyper-networks is used to represent multi-modal and composite interactions between data points. In such networks, several different types of nodes form a hyperedge. Heterogeneous hyper-network embedding learns a distributed node representation under such complex interactions while preserving the network structure. However, this is a challenging task due to the multiple modalities and composite interactions. In this
study, a deep approach is proposed to embed heterogeneous attributed
hyper-networks with complicated and non-linear node relationships. In particular, a fully-connected and graph convolutional layers are designed to project different
types of nodes into a common low-dimensional space, a
tuple-wise similarity function is proposed to preserve the network structure, and a ranking based loss function is used to improve the similarity scores of
hyperedges in the embedding space.

# Usuage
python main.py data_path output_path learning_rate epoch_number

# Input
Nxd node features for each type, known edges, sampled unknown hyperedges in list format, sparse normalized adjacency matrix. This implementation is for a network for three different types.

# Reference
Inci M. Baytas, Cao Xiao, Fei Wang, Anil K. Jain, Jiayu Zhou, "Heterogeneous Hyper-Network Embedding", ICDM 2018 (accepted).
