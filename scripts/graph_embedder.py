import glob
import tensorflow as tf
import numpy as np
import pandas as pd
import stellargraph as sg
import networkx as nx
from sklearn.cluster import AffinityPropagation


class Graph_Embedder:
    def __init__(self,clients_folders) -> None:
        self.clients_folders = clients_folders
        self.clients_graphs = {}
        self.graphs = []
        for client_folder in self.clients_folders:
            models_list = self.__browse_dir(client_folder)
            for model in models_list:
                df_data = self.__cnn_to_weights_df(model)
                df = self.__cnn_to_df(model)
                cnn_graph = sg.StellarGraph(df_data, df)
                self.clients_graphs[model] = cnn_graph
                self.graphs.append(cnn_graph)

    def __browse_dir(self,path_name):
        return [model for model in glob.glob(path_name + "*.h5")]

    def __graph_distance(self,graph1, graph2):
        spec1 = nx.laplacian_spectrum(graph1.to_networkx(feature_attr=None))
        spec2 = nx.laplacian_spectrum(graph2.to_networkx(feature_attr=None))
        k = min(len(spec1), len(spec2))
        return np.linalg.norm(spec1[:k] - spec2[:k])

    def __cnn_to_weights_df(self,cnn_model):
        weights = {}
        loaded_model = tf.keras.models.load_model(cnn_model)
        for layer in loaded_model.layers:
            if (layer.trainable_weights):
                weights[layer.name] = [np.mean(layer.get_weights()[0]),np.mean(layer.get_weights()[1])]
            else:
                weights[layer.name] = [0,0]
        df_data = pd.DataFrame.from_dict(data = weights, orient='index', columns=['W', 'B'])    
        return df_data


    def __cnn_to_df(self, cnn_model):
        loaded_model = tf.keras.models.load_model(cnn_model)
        layers = []
        source = []
        destination = []
        for layer in loaded_model.layers:
            layers.append(layer.name)
        for i in range(0,len(layers)-1):
            source.append(layers[i])
            destination.append(layers[i+1])
        df = pd.DataFrame(list(zip(source, destination)),columns =['source', 'target'])    
        return df


    def generate_embeddings(self, graphs):
        generator = sg.mapper.PaddedGraphGenerator(graphs)
        gc_model = sg.layer.GCNSupervisedGraphClassification(
            [64, 32], ["relu", "relu"], generator, pool_all_layers=True
        )
        inp1, out1 = gc_model.in_out_tensors()
        inp2, out2 = gc_model.in_out_tensors()

        vec_distance = tf.norm(out1 - out2, axis=1)
        pair_model = tf.keras.Model(inp1 + inp2, vec_distance)
        embedding_model = tf.keras.Model(inp1, out1)
        #embedding_model.summary()

        graph_idx = np.random.RandomState(0).randint(len(graphs), size=(100, 2))
        targets = [self.__graph_distance(graphs[left], graphs[right]) for left, right in graph_idx]
        train_gen = generator.flow(graph_idx, batch_size=10, targets=targets)
        pair_model.compile(tf.keras.optimizers.Adam(1e-2), loss="mse")

        history = pair_model.fit(train_gen, epochs=500, verbose=0)
        #sg.utils.plot_history(history)
        embeddings = embedding_model.predict(generator.flow(graphs))
        #print('Embeddings are generated ... Shape:{0}',embeddings.shape)
        return embeddings

    def cluster_embeddings(self, embeddings):
        #print('Affinity Propagation Clustering...')
        affinityPropagation = AffinityPropagation(random_state=1234).fit(embeddings)
        result = list(zip(self.clients_graphs.keys(),affinityPropagation.labels_))
        with open('embeddings_clustering.csv', 'w') as txt_file :
            for item in result:
                txt_file.write("{0},{1}\n".format(item[0],item[1]))

    
    
    
    #result = list(zip(clients_graphs.keys(),affinityPropagation.labels_))

'''
clients_folders= ['./clients/1/','./clients/2/']
GE = Graph_Embedder(clients_folders)

graphs_embeddings = GE.generate_embeddings(GE.graphs)
cluster_labels = GE.cluster_embeddings(graphs_embeddings)
result = list(zip(GE.clients_graphs.keys(),cluster_labels))
print(result)
'''
