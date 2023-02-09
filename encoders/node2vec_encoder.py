from node2vec import Node2Vec
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
import os

class FastEmbedder():
  def __init__(self, keyed_vectors):
    self.keyed_vectors = keyed_vectors

  def _embed(self, right_nodes, left_nodes):
    return self.keyed_vectors[right_nodes], self.keyed_vectors[left_nodes]

  def __getitem__(self, edges):
    if edges.ndim > 1:
      right_nodes = edges[:,0]
      left_nodes = edges[:,1]
    else:
      right_nodes = edges[0]
      left_nodes = edges[1]
    return self._embed(right_nodes, left_nodes)


class Node2VecEncode():
    def __init__(
        self,
        graph,
        input_size,
        walk_length,
        p,
        q,
        directory,
        species,
        seed,
        ):
        '''
        Node2vec encoder
        Parameters:
            graph (deepsnap.graph.Graph): DeepSnap training graph used for embedding the nodes
            input_size (int): size of the input vector
            walk_length (int): length of the random walks
            p (float): return parameter
            q (float): inout parameter
            directory (str): the directory where the node2vec embeddings are stored
            species (str): the species evaluated
            seed (int): the seed used to create dataset

        '''
         #create directory
        os.makedirs(directory, exist_ok=True) 

        #get node2vec directory
        n2v_dir = directory + f"/{species}_embedding_model_dim:{input_size}_len:{walk_length}_p:{p}_q:{q}_seed:{seed}"

        #try to load the model from the directory if it already exists 
        try:
            model = Word2Vec.load(n2v_dir)

        #if the model doesn't exist then create the model and save it
        except:
            G = nx.Graph()
            G.add_edges_from(self.get_edge_list(graph))

            #generate walks
            node2vec = Node2Vec(G, dimensions=input_size, walk_length=walk_length, num_walks=50, workers=1, p=p, q=q)

            #train node2vec model
            n2w_model = node2vec.fit(window=7, min_count=1)

            #save model for later use
            n2w_model.save(n2v_dir)

            #get word2vec model
            model = Word2Vec.load(n2v_dir)

        #instanciate the edge embedder
        self.embedder = FastEmbedder(keyed_vectors=model.wv)

        #get edges of the graph
        edges = graph.edge_index.numpy()

        #get the nodes contained in the message passing graph
        nodes = set(np.hstack([edges[0], edges[1]]).tolist())

        #get the nodes not included in the message passing graph
        nodes_not_seen = [node for node in graph.node_label_index.numpy() if node not in nodes]
        assert len(nodes_not_seen) + len(nodes) == len(graph.node_label_index.numpy())

        #add the missing nodes to the keyed vectors (all 0's)
        self.embedder.keyed_vectors.add_vectors(nodes_not_seen, [np.zeros(input_size, dtype=np.float32) for _ in range(len(nodes_not_seen))])


    def get_edge_list(self, graph):
        '''
        Get the edge list for the deepsnap graph
        Parameter:
            graph (deepsnap.graph.Graph): DeepSnap training graph used for embedding the nodes
        Return:
            List(Tuple(str, str)): edge list
        '''
        return [tuple(e) for e in graph.edge_index.numpy().T]

    
    def __call__(self, e):
        return self.embedder[e]