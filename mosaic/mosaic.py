from platform import node
from typing import Set, List, Optional, Tuple
import math, random
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
from scipy.sparse import csr_matrix
import time

# visualisation
import networkx as nx
import matplotlib.pyplot as plt

ROOT_LABEL = "ROOT"

class Node:
    def __init__(self, phrase : str, parent = None):
        self.phrase = phrase
        self.parent = parent
        self.children = {}
        self.generative_links = []

    def __contains__(self, utterance : List[str]) -> bool:
        if len(utterance) == 0:
            return True
        if len(utterance) == 1:
            return utterance[0] in self.children
        if not utterance[0] in self.children:
            return False
        return utterance[1:] in self.children[utterance[0]] 

    def has_children(self) -> bool:
        if self.children:
            return True
        return False

    def has_child(self, child : str) -> bool:
        return child in self.children

    def get_children(self):
        return list(self.children.values())

    def __str__(self) -> str:
        return '({})'.format(','.join([child.phrase+str(child) for child in self.get_children()])) if len(self.children) > 0 else ""

    def add_node(self, phrase : str):
        if not phrase in self.children:
            new_node = Node(phrase, self)
            self.children[phrase] = new_node

    def get_contexts(self) -> Set[str]:
        """ Returns the words that directly precede or follow this node """
        contexts = [child.phrase for child in self.get_children()]
        if self.parent and self.parent.phrase != ROOT_LABEL:
            backward_context = self.parent.phrase
            contexts.append(backward_context)
        return set(contexts)

    def get_max_phrase_length(self, depth=0) -> int:
        if len(self.children) == 0:
            return depth
        return max([child.get_max_phrase_length(depth+1) for child in self.children.values()])

    def get_total_nodes(self) -> int: 
        return sum([child.get_total_nodes() for child in self.children.values()]) + 1

    def get_total_phrases(self) -> int:
        if not self.children:
            return 1
        return sum([child.get_total_phrases() for child in self.children.values()])

    def get_total_generative_links(self) -> int:
        return sum([child.get_total_generative_links() for child in self.children.values()]) + len(self.generative_links)

    def get_all_nodes(self):
        nodes = [self]
        for child in self.children.values():
            nodes.extend(child.get_all_nodes())
        return nodes

class WordNetwork:
    def __init__(self, verbose : bool = False, calculate_ncp = True, corpus_size : int = math.inf, m = 20):
        self.root = Node('ROOT')
        self.verbose = verbose
        self.calculate_ncp = calculate_ncp
        self.corpus_size = corpus_size
        self.num_utterances_seen = 0
        self.m = m

    def get_node_creation_probability(self, distance_to_end_of_utterance):
        if self.calculate_ncp:
            ncp = pow((1 / (1+math.exp(self.m - self.num_utterances_seen/self.corpus_size))), math.sqrt(distance_to_end_of_utterance))
            return ncp
        else:
            return 1

    def __str__(self) -> str:
        return str(self.root)

    def add_primitive_node(self, word : str):
        if random.random() <= self.get_node_creation_probability(1):
            if self.verbose:
                print("adding primitive node: ", word)
            self.root.add_node(word)

    def add_utterance(self, utterance : List[str]):
        if self.verbose:
            print("adding phrase:", utterance)
        current_node = self.root
        for position, phrase in enumerate(utterance):
            distance = len(utterance) - position
            if not current_node.has_child(phrase):
                if random.random() <= self.get_node_creation_probability(distance):
                    current_node.add_node(phrase)
                else:
                    break
            current_node = current_node.children[phrase]

    def process_utterances(self, utterances : List[List[str]]):
        for utterance in utterances:
            for i in range(len(utterance)):
                sub_utterance = utterance[-i-1:]
                # create primitive node for unseen words
                if not self.root.has_child(sub_utterance[0]):
                    self.add_primitive_node(sub_utterance[0])
                    break
                # create phrase encoding for unseen phrase
                elif not sub_utterance in self.root:
                    self.add_utterance(sub_utterance)
                    break
            
            self.num_utterances_seen += 1
        self.make_generative_links()

    def visualise(self):
        g = nx.DiGraph()
 
        def add_branches(parent : Node, full_phrase = ""):
            for child in parent.get_children():
                phrase = child.phrase
                g.add_edge(full_phrase, full_phrase + " " + phrase, label=phrase)
                add_branches(child, full_phrase + " " + phrase)
            #for generative_link in parent.generative_links:
            #    g.add_edge(parent.phrase, full_phrase + " " + generative_link.phrase, weight=3)
        
        add_branches(self.root)

        pos = nx.circular_layout(g)
        nx.draw_networkx(g, pos, with_labels = True)
        plt.plot()

    def get_mean_phrase_length(self) -> float : 
        lengths = []

        def get_lengths(parent : Node, depth):
            if not parent.has_children():
                lengths.append(depth)
            for child in parent.get_children():
                get_lengths(child, depth+1)

        get_lengths(self.root, 0)

        return sum(lengths) / len(lengths) if len(lengths) > 0 else 0

    def print_stats(self):
        print("Number of nodes:", self.root.get_total_nodes())
        print("Number of phrases:", self.root.get_total_phrases())
        print("Mean phrase length:",self.get_mean_phrase_length())
        print("Maximum phrase length:", self.root.get_max_phrase_length())
        print("Number of generative links:", self.root.get_total_generative_links())

    def make_generative_links(self):
        nodes = self.root.get_all_nodes()[1:] # don't link to root

        contexts = []
        nodes_with_contexts = []
        lengths = []
        num_nodes = 0

        vocab = {}
        word_key = 0

        # Get all nodes with contexts that contain more than 2 words and clear current generative links
        # Contexts are converted to lists of vocab indices for efficiency in the overlap calculation
        # We also store the lengths of the contexts to save on n^2 length calculations later
        for i, node in enumerate(nodes):
            node.generative_links = []
            context = node.get_contexts()
            length = len(context)
            if length <= 2:
                continue
            for word in context:
                if not word in vocab:
                    vocab[word] = word_key
                    word_key += 1
            contexts.append(set([vocab[word] for word in context]))
            nodes_with_contexts.append(node)
            lengths.append(length)
            num_nodes += 1

        for i in tqdm (range (num_nodes), desc="Processing Nodes..."):
            node_a = nodes_with_contexts[i]
            length_a = lengths[i]
            context_a = contexts[i]
            # Add link for every node whose context overlaps at least 20%
            # node_a.generative_links = [nodes_with_contexts[j] for j in range(num_nodes)
            #    if node_a != nodes_with_contexts[j] and 
            #    len(context_a & contexts[j]) > (length_a + lengths[j]) * 0.2] 
            for j in range(num_nodes):
                node_b = nodes_with_contexts[j]
                if node_a == node_b:
                    continue
                context_b = contexts[j]
                length_b = lengths[j]
                if len(context_a & context_b) > (length_a + length_b) * 0.2:
                    node_a.generative_links.append(node_b)

from dataset import DataSet

data = DataSet("data/corpora/aochildes_under3_adult.txt")
print("Number of utterances:", data.size)
random.seed(0)
network = WordNetwork(verbose=False, calculate_ncp=True, corpus_size=data.size, m=20)
for i in range(30):
    if i > 15:
        print("Iteration:",i)
    network.process_utterances(data.utterances)
    if i > 15:
        network.print_stats()
        print()
