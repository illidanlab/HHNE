# Inci M. Baytas
# Usuage: python main.py data_path output_path 1e-4 50

import tensorflow as tf
import numpy as np
import cPickle
import sys



def save_pkl(path,obj):
    with open(path, 'w') as f:
        cPickle.dump(obj,f)

def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
        return obj


from HNE import HNE

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def get_tuple_sim(E,dim_1,dim_2,dim_3,d1_emb,d2_emb,d3_emb):
    tuple_sim = np.zeros(len(E))
    for i in range(len(E)):
        s = E[i].sum(axis=1)
        edge = E[i][s != 0]
        p = edge[edge[:, 0] == dim_1][:, 1]
        d = edge[edge[:, 0] == dim_2][:, 1]
        a = edge[edge[:, 0] == dim_3][:, 1]
        p_emb = d1_emb[p.astype(int)]
        d_emb = d2_emb[d.astype(int)]
        a_emb = d3_emb[a.astype(int)]
        emb_mat = np.concatenate([p_emb, d_emb, a_emb], axis=0)
        sim = np.matmul(emb_mat, np.transpose(emb_mat))
        diagonal = np.diag(np.diag(sim))
        upper_tri = np.triu(sim)
        sim = upper_tri - diagonal
        sim_vals = sim[np.abs(sim) > 0]
        tuple_sim[i] = sim_vals.mean()
    return tuple_sim

def get_accuracy(pos_sim,neg_sim):
    number_positives = len(pos_sim)
    number_negatives = len(neg_sim)
    edge_similarities = np.zeros(number_positives+number_negatives)
    edge_labels = np.zeros(number_positives+number_negatives)
    count = 0
    for i in range(len(pos_sim)):
        edge_similarities[count] = pos_sim[i]
        edge_labels[count] = 1
        count += 1
    for i in range(len(neg_sim)):
        edge_similarities[count] = neg_sim[i]
        edge_labels[count] = 0
        count += 1
    sorted_edge_labels = edge_labels[np.argsort(-edge_similarities)]
    learned_labels = np.zeros(number_positives+number_negatives)
    learned_labels[:number_positives] = 1
    ap = average_precision(sorted_edge_labels)
    prec = precision_at_k(sorted_edge_labels, number_positives)
    return ap,prec


def train_test(data_path,output_path,learning_rate,number_epochs):
    d1_fea = load_pkl(data_path + "d1_fea.pkl")
    d2_fea = load_pkl(data_path + "d2_fea.pkl")
    d3_fea = load_pkl(data_path + "d3_fea.pkl")

    adj = load_pkl(data_path + "adj.pkl")
    adj = adj.todense()

    E_pos_train = load_pkl(data_path + "train_E_pos.pkl")
    E_neg_train = load_pkl(data_path + "train_E_neg.pkl")
    E_pos_test = load_pkl(data_path + "test_E_pos.pkl")
    E_neg_test = load_pkl(data_path + "test_E_neg.pkl")


    num_batch = len(E_pos_train)

    d1 = d1_fea.shape[1]
    d2 = d2_fea.shape[1]
    d3 = d3_fea.shape[1]

    K = 3

    r1, r2, r3 = 128, 64, 32
    n1 = len(d1_fea)
    n2 = len(d2_fea)
    n3 = len(d3_fea)

    hne = HNE(K, d1, d2, d3, n1, n2, n3, r1, r2, r3)
    cost = hne.get_cost()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # RMSPropOptimizer

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for e in range(number_epochs):
            Cost = np.zeros(num_batch)
            for i in range(num_batch):
                _, Cost[i] = sess.run([optimizer, cost], feed_dict={hne.pos_edges: E_pos_train[i], \
                                                                 hne.neg_edges: E_neg_train[i], \
                                                                 hne.d1_fea: d1_fea, \
                                                                 hne.d2_fea: d2_fea, \
                                                                 hne.d3_fea: d3_fea,
                                                                 hne.A: adj})
            print("Epoch %d, Cost = %.5f" %(e, Cost.mean()))
        print("Training is over!")

        d1_emb, d2_emb, d3_emb = sess.run(hne.get_embs(), feed_dict={hne.pos_edges: E_pos_test, \
                                                                 hne.neg_edges: E_neg_test, \
                                                                 hne.d1_fea: d1_fea, \
                                                                 hne.d2_fea: d2_fea, \
                                                                 hne.d3_fea: d3_fea,
                                                                 hne.A: adj})
        save_pkl(output_path + "d1_emb.pkl", d1_emb)
        save_pkl(output_path + "d2_emb.pkl", d2_emb)
        save_pkl(output_path + "d3_emb.pkl", d3_emb)

        pos_sim, neg_sim = sess.run(hne.edge_similarity(), feed_dict={hne.pos_edges: E_pos_test, \
                                                                hne.neg_edges: E_neg_test, \
                                                                hne.d1_fea: d1_fea, \
                                                                hne.d2_fea: d2_fea, \
                                                                hne.d3_fea: d3_fea,
                                                                hne.A: adj})


        pos_sim = np.reshape(pos_sim, [len(pos_sim)])
        neg_sim = np.reshape(neg_sim, [len(neg_sim)])
        AP, PREC = get_accuracy(pos_sim, neg_sim)

        print("Hyperedge detection by the proposed similarity")
        print("Prec:%.4f" % (PREC))
        print("AP:%.4f" % (AP))

        pos_sim_external = get_tuple_sim(E_pos_test,d1,d2,d3,d1_emb,d2_emb,d3_emb)
        neg_sim_external = get_tuple_sim(E_neg_test, d1, d2, d3, d1_emb, d2_emb, d3_emb)
        AP_external, PREC_external = get_accuracy(pos_sim_external, neg_sim_external)

        print("Hyperedge detection by the baseline similarity computation")
        print("Prec:%.4f" % (PREC_external))
        print("AP:%.4f" % (AP_external))


def main(argv):

    data_path = str(sys.argv[1])
    output_path = str(sys.argv[2])

    learning_rate = float(sys.argv[3])
    num_epochs = int(sys.argv[4])

    train_test(data_path, output_path, learning_rate, num_epochs)



if __name__ == "__main__":
    main(sys.argv[1:])



