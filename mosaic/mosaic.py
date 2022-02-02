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
    def __init__(self, phrase : List[str], parent = None):
        self.phrase = ' '.join(phrase)
        self.parent = parent
        self.branches = {}
        self.generative_links = []

    def __contains__(self, phrase : List[str]) -> bool:
        if len(phrase) == 0:
            return True
        if len(phrase) == 1:
            return phrase[0] in self.branches
        if not phrase[0] in self.branches:
            return False
        return phrase[1:] in self.branches[phrase[0]] 

    def has_branch(self, branch : str) -> bool:
        return branch in self.branches

    def __str__(self) -> str:
        return '({})'.format(','.join([key+str(self.branches[key]) for key in self.branches])) if len(self.branches) > 0 else ""

    def add_phrase(self, phrase : List[str], position : int = 0):
        if not phrase[position] in self.branches:
            new_node = Node(phrase[:position+1])
            self.branches[phrase[position]] = new_node
        if position == len(phrase)-1:
            return
        else:
            self.branches[phrase[position]].add_phrase(phrase, position+1)

    def add_node(self, branch : str, phrase : List[str]):
        if not branch in self.branches:
            new_node = Node(phrase, self)
            self.branches[branch] = new_node

    def get_contexts(self) -> Set[str]:
        """ Returns the words that directly precede or follow this node """
        contexts = list(self.branches.keys())
        if self.parent and self.parent.phrase != ROOT_LABEL:
            backward_context = self.parent.phrase.split(' ')[-1]
            contexts.append(backward_context)
        return set(contexts)

    def get_max_phrase_length(self, depth=0) -> int:
        if len(self.branches) == 0:
            return depth
        return max([child.get_max_phrase_length(depth+1) for child in self.branches.values()])

    def get_total_nodes(self) -> int: 
        return sum([child.get_total_nodes() for child in self.branches.values()]) + 1

    def get_total_phrases(self) -> int:
        if not self.branches:
            return 1
        return sum([child.get_total_phrases() for child in self.branches.values()])

    def get_total_generative_links(self) -> int:
        return sum([child.get_total_generative_links() for child in self.branches.values()]) + len(self.generative_links)

    def get_all_nodes(self):
        nodes = [self]
        for child in self.branches.values():
            nodes.extend(child.get_all_nodes())
        return nodes

class WordNetwork:
    def __init__(self, verbose : bool = False, calculate_ncp = True, corpus_size : int = math.inf, m = 20):
        self.root = Node(['ROOT'])
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
            self.root.add_node(word, [word])

    def add_phrase(self, phrase : List[str]):
        if self.verbose:
            print("adding phrase:", phrase)
        current_node = self.root
        for position, word in enumerate(phrase):
            distance = len(phrase) - position
            if not current_node.has_branch(word):
                if random.random() <= self.get_node_creation_probability(distance):
                    current_node.add_node(word, phrase[:position+1])
                else:
                    break
            current_node = current_node.branches[word]

    def process_utterances(self, utterances : List[List[str]]):
        for utterance in utterances:
            for i in range(len(utterance)):
                phrase = utterance[-i-1:]
                # create primitive node for unseen words
                if not self.root.has_branch(phrase[0]):
                    self.add_primitive_node(phrase[0])
                    break
                # create phrase encoding for unseen phrase
                elif not phrase in self.root:
                    self.add_phrase(phrase)
                    break
            
            self.num_utterances_seen += 1
        self.make_generative_links()

    def visualise(self):
        g = nx.DiGraph()
 
        def add_branches(parent : Node):
            for branch in parent.branches:
                child = parent.branches[branch]
                g.add_edge(parent.phrase, child.phrase, label=branch)
                add_branches(child)
            for generative_link in parent.generative_links:
                g.add_edge(parent.phrase, generative_link.phrase, weight=3)
        
        add_branches(self.root)

        pos = nx.circular_layout(g)
        nx.draw_networkx(g, pos, with_labels = True)
        plt.plot()

    def get_mean_phrase_length(self) -> float : 
        lengths = []

        def get_lengths(parent : Node, depth):
            if len(parent.branches) == 0:
                lengths.append(depth)
            for child in parent.branches.values():
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
            node_a.generative_links = [nodes_with_contexts[j] for j in range(num_nodes) if len(context_a & contexts[j]) > (length_a + lengths[j]) * 0.2]   

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
