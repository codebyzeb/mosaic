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

    def get_max_utterance_length(self, depth=0) -> int:
        if len(self.children) == 0:
            return depth
        return max([child.get_max_utterance_length(depth+1) for child in self.children.values()])

    def get_total_nodes(self) -> int: 
        return sum([child.get_total_nodes() for child in self.children.values()]) + 1

    def get_total_utterances(self) -> int:
        if not self.children:
            return 1
        return sum([child.get_total_utterances() for child in self.children.values()])

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
        self.make_generative_links_using_joint_contexts()

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

    def get_mean_utterance_length(self) -> float : 
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
        print("Number of utterances:", self.root.get_total_utterances())
        print("Mean utterance length:",self.get_mean_utterance_length())
        print("Maximum utterance length:", self.root.get_max_utterance_length())
        print("Number of generative links:", self.root.get_total_generative_links())

    def make_generative_links_using_node_contexts(self):
        """ Creates links between nodes that share similar contexts (compares each node's context to every other node's context).
        E.g. the context of each occurrence of "him" is treated separately, not combined into one joint context. """

        nodes = self.root.get_all_nodes()[1:] # don't link to root

        contexts = []
        nodes_with_contexts = []
        lengths = []
        num_nodes = 0

        vocab = {}
        phrase_key = 0

        # Get all nodes with contexts that contain more than 2 words and clear current generative links
        # Contexts are converted to lists of vocab indices for efficiency in the overlap calculation
        # We also store the lengths of the contexts to save on n^2 length calculations later
        for i, node in enumerate(nodes):
            node.generative_links = []
            context = node.get_contexts()
            length = len(context)
            if length <= 2:
                continue
            for phrase in context:
                if not phrase in vocab:
                    vocab[phrase] = phrase_key
                    phrase_key += 1
            contexts.append(set([vocab[phrase] for phrase in context]))
            nodes_with_contexts.append(node)
            lengths.append(length)
            num_nodes += 1

        for i in tqdm (range (num_nodes), desc="Processing Nodes..."):
            node_a = nodes_with_contexts[i]
            length_a = lengths[i]
            context_a = contexts[i]
            # Add link for every node whose context overlaps at least 20%
            node_a.generative_links = [nodes_with_contexts[j] for j in range(num_nodes)
               if node_a != nodes_with_contexts[j] and 
               len(context_a & contexts[j]) > (length_a + lengths[j]) * 0.2] 

    def make_generative_links_using_joint_contexts(self):
        """ Creates links between nodes that share similar contexts by combining the contexts of ALL occurrences of the phrase
        encoded by that node within the network. E.g. the joint contexts (words that appear before and after) of every occurrence
        of "her" is compared to the joint contexts of every occurrence of "him" """

        nodes = self.root.get_all_nodes()[1:] # don't link to root

        phrase_to_context = {}
        phrase_to_nodes = {}
        phrase_to_number = {}
        phrase_key = 0
        num_unique_phrases = 0

        for i, node in enumerate(nodes):
            # Clear all previous generative links
            node.generative_links = []
            context = node.get_contexts()

            if len(context) < 2:
                continue
            
            for phrase in context:
                if not phrase in phrase_to_number:
                    phrase_to_number[phrase] = phrase_key
                    phrase_key += 1
            
            if not node.phrase in phrase_to_context:
                phrase_to_context[node.phrase] = set()
                phrase_to_nodes[node.phrase] = []
                num_unique_phrases += 1
            
            phrase_to_nodes[node.phrase].append(node)
            # Contexts are sets, so we're only looking at types, not tokens
            phrase_to_context[node.phrase].update([phrase_to_number[phrase] for phrase in context])
        
        phrase_to_context_length = {}
        for phrase in phrase_to_context:
            phrase_to_context_length[phrase] = len(phrase_to_context[phrase])

        for phrase in tqdm (phrase_to_context, desc="Processing Phrases..."):
            context = phrase_to_context[phrase]
            context_length = len(context)
            # Get all phrases whose context overlaps at least 20%
            similar_phrases = [other_phrase for other_phrase in phrase_to_context
                               if other_phrase != phrase and
                               len(context & phrase_to_context[other_phrase]) > (context_length + phrase_to_context_length[other_phrase]) * 0.2]
            # Add links between similar phrases
            for other_phrase in similar_phrases:
                for node_a in phrase_to_nodes[phrase]:
                    node_a.generative_links = phrase_to_nodes[other_phrase]
