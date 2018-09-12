import tensorflow as tf


class HNE(object):
    def init_weights(self, input_dim, output_dim, name, reg=None):
        var = tf.get_variable(name, shape=[input_dim, output_dim],
                               initializer=tf.random_normal_initializer(0.0, 0.1), regularizer=reg)
        return var

    def init_bias(self, output_dim, name):
        var =  tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(0.0))
        return var

    def __init__(self, K, d1, d2, d3, n1, n2, n3, r1, r2, r3):

        self.K = K  # Number of types
        self.d1 = d1  # dimensionality of type 1
        self.d2 = d2  # dimensionality of type 2
        self.d3 = d3  # dimensionality of type 3
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.r1 = r1  # common initial embedding space
        self.r2 = r2  # second layer common embedding dimensionality
        self.r3 = r3 # common embedding dimensionality


        # number of hyperedges x number of vertices x input dim (max dim)
        self.pos_edges = tf.placeholder(tf.float32, shape=[None, None, None])  # None, None
        self.neg_edges = tf.placeholder(tf.float32, shape=[None, None, None])  # None, None
        self.d1_fea = tf.placeholder(tf.float32, shape=[None,None])
        self.d2_fea = tf.placeholder(tf.float32, shape=[None,None])
        self.d3_fea = tf.placeholder(tf.float32, shape=[None,None])

        self.A = tf.placeholder(tf.float32, shape=[None,None])


        self.WE1 = self.init_weights(self.d1, self.r1, name='Embedding_1_1', reg=None)
        self.bE1 = self.init_bias(self.r1, name='Embedding_bias_1_1')

        self.WE2 = self.init_weights(self.d2, self.r1, name='Embedding_1_2', reg=None)
        self.bE2 = self.init_bias(self.r1, name='Embedding_bias_1_2')

        self.WE3 = self.init_weights(self.d3, self.r1, name='Embedding_1_3', reg=None)
        self.bE3 = self.init_bias(self.r1, name='Embedding_bias_1_3')

        self.Wgcnn1 = self.init_weights(self.r1, self.r2, name='GCNN1', reg=None)
        self.Wgcnn2 = self.init_weights(self.r2, self.r3, name='GCNN2', reg=None)


        # Weights of similarity function
        self.U1 = self.init_weights(self.r3, self.r3, name='Sim1', reg=None)
        self.U2 = self.init_weights(self.r3, self.r3, name='Sim2', reg=None)
        self.U3 = self.init_weights(self.r3, self.r3, name='Sim3', reg=None)

        self.W_lin = self.init_weights(self.r3, K, name='Sim', reg=None)
        self.b_lin = self.init_bias(self.K, name='Sim_bias')
        self.V = self.init_weights(self.K, 1, name='Sim_out', reg=None)


    def get_embedding_d1(self, v):
        return tf.nn.tanh(tf.matmul(v, self.WE1) + self.bE1)

    def get_embedding_d2(self, v):
        return tf.nn.tanh(tf.matmul(v, self.WE2) + self.bE2)

    def get_embedding_d3(self, v):
        return tf.nn.tanh(tf.matmul(v, self.WE3) + self.bE3)

    def GCNN(self, adj,x):
        layer1 = tf.nn.tanh(tf.matmul(tf.matmul(adj,x),self.Wgcnn1))
        layer2 = tf.matmul(tf.matmul(adj,layer1),self.Wgcnn2)
        return layer2

    def get_embs(self):
        d1_eb1 = self.get_embedding_d1(self.d1_fea)
        d2_eb1 = self.get_embedding_d2(self.d2_fea)
        d3_eb1 = self.get_embedding_d3(self.d3_fea)
        X = tf.concat([d1_eb1,d2_eb1,d3_eb1],axis=0)
        emb = self.GCNN(self.A,X)
        d1_eb = tf.slice(emb, [0,0], [self.n1, self.r3])
        d2_eb = tf.slice(emb,[self.n1,0],[self.n2,self.r3])
        d3_eb = tf.slice(emb, [(self.n1 + self.n2), 0], [self.n3, self.r3])

        return d1_eb, d2_eb, d3_eb

    def get_bilinear_sim(self, indices):
        zero = tf.constant(0, dtype=tf.float32)
        s = tf.reduce_sum(indices, axis=1)
        bool_mask = tf.not_equal(s, zero)
        indices = tf.boolean_mask(indices, bool_mask)
        emb_indices = tf.slice(indices, [0, 1], [tf.shape(indices)[0], 1])

        dims = tf.slice(indices, [0, 0], [tf.shape(indices)[0], 1])
        bool_d1 = tf.equal(dims, self.d1)
        bool_d2 = tf.equal(dims, self.d2)
        bool_d3 = tf.equal(dims, self.d3)
        d1_indices = tf.cast(tf.boolean_mask(emb_indices, bool_d1), tf.int32)
        d2_indices = tf.cast(tf.boolean_mask(emb_indices, bool_d2), tf.int32)
        d3_indices = tf.cast(tf.boolean_mask(emb_indices, bool_d3), tf.int32)

        drug = tf.gather(self.d1_embs, d1_indices)
        indi = tf.gather(self.d2_embs,d2_indices)
        adr = tf.gather(self.d3_embs, d3_indices)

        emb_matrix = tf.concat([drug,indi,adr],0)


        b1 = tf.matmul(tf.matmul(drug, self.U1), tf.transpose(indi))
        b1 = tf.reduce_sum(b1)
        b1 = tf.reshape(b1,[1,1])
        b2 = tf.matmul(tf.matmul(drug, self.U2), tf.transpose(adr))
        b2 = tf.reduce_sum(b2)
        b2 = tf.reshape(b2, [1, 1])
        b3 = tf.matmul(tf.matmul(indi, self.U3), tf.transpose(adr))
        b3 = tf.matrix_band_part(b3, 0, -1)
        b3 = tf.reduce_sum(b3)
        b3 = tf.reshape(b3, [1, 1])

        B = tf.concat([b1,b2,b3],1)

        B2 = tf.reduce_mean(tf.matmul(emb_matrix, self.W_lin),0)


        sim = tf.matmul(tf.nn.tanh(B + B2 + self.b_lin), self.V)

        return sim


    def edge_similarity(self):
        self.d1_embs, self.d2_embs, self.d3_embs = self.get_embs()

        pos_sim = tf.map_fn(self.get_bilinear_sim, self.pos_edges, name='pos_edge_similarities')
        neg_sim = tf.map_fn(self.get_bilinear_sim, self.neg_edges, name='neg_edge_similarities')
        return pos_sim, neg_sim


    def get_cost(self):

        Se, Se0 = self.edge_similarity()
        sum_Se0 = tf.reduce_mean(Se0)
        term1 = tf.reduce_mean(tf.log(1 + tf.exp(sum_Se0 - Se)))
        cost = term1
        return cost





