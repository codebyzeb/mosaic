import math, random
from typing import Set, List, Optional, Tuple
from tqdm import tqdm

# visualisation
import networkx as nx
import matplotlib.pyplot as plt

from params import Params

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
        return "({})".format(",".join([child.phrase+str(child) for child in self.get_children()])) if len(self.children) > 0 else ""

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
    def __init__(self, verbose : bool = False, params : Params = None):
        self.root = Node("ROOT")
        self.num_utterances_seen = 0
        self.verbose = verbose

        self.params = params if params else Params()

    def get_node_creation_probability(self, distance_to_end_of_utterance):
        if self.params.calculate_ncp:
            ncp = pow((1 / (1+math.exp(self.params.m - self.num_utterances_seen/self.params.corpus_size))), math.sqrt(distance_to_end_of_utterance))
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
        # If the corpus size hasn't been set in advance, set it now
        if not self.params.corpus_size:
            self.params.corpus_size = len(utterances)
            if self.verbose:
                print("Setting corpus_size parameter to {}".format(self.params.corpus_size))
        
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
        if self.params.generative_links_using_joint_contexts:
            self.make_generative_links_using_joint_contexts()
        else:
            self.make_generative_links_using_node_contexts()

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

    def print_stats(self, include_generated_utterances_in_stats=True):
        num_nodes = self.root.get_total_nodes()
        num_generative_links = self.root.get_total_generative_links()
        if not include_generated_utterances_in_stats:
            num_utterances = self.root.get_total_utterances()
            mean_utterance_length = self.get_mean_utterance_length()
            proportion_novel_utterances = 0
        else:
            learned_utterances = self.get_learned_utterances()
            generated_utterances = self.get_generated_utterances()
            all_utterances = learned_utterances | generated_utterances
            num_utterances = len(all_utterances)
            mean_utterance_length = sum([utterance.count(" ") + 1 for utterance in all_utterances]) / num_utterances if num_utterances > 0 else 0
            proportion_novel_utterances = (num_utterances - len(learned_utterances)) / num_utterances if num_utterances > 0 else 0

        print("Number of nodes:", num_nodes)
        print("Number of generative links:", num_generative_links)
        print("Number of utterances:", num_utterances)
        print("Mean utterance length: {0:.3g}".format(mean_utterance_length))
        print("Proportion of novel utterances: {0:.3g}".format(proportion_novel_utterances))

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

        for i in tqdm (range (num_nodes), desc="Calculating Generative Links for Nodes...", disable=self.params.hide_progress_bar):
            node_a = nodes_with_contexts[i]
            length_a = lengths[i]
            context_a = contexts[i]
            # Add link for every node whose context overlaps at least 20%
            node_a.generative_links = [nodes_with_contexts[j] for j in range(num_nodes)
               if node_a != nodes_with_contexts[j] and 
               len(context_a & contexts[j]) > (length_a + lengths[j]) * self.params.overlap_threshold] 

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

        for phrase in tqdm (phrase_to_context, desc="Processing Phrases...", disable=self.params.hide_progress_bar):
            context = phrase_to_context[phrase]
            context_length = len(context)
            # Get all phrases whose context overlaps at least 20%
            similar_phrases = [other_phrase for other_phrase in phrase_to_context
                               if other_phrase != phrase and
                               len(context & phrase_to_context[other_phrase]) > (context_length + phrase_to_context_length[other_phrase]) * self.params.overlap_threshold]
            # Add links between similar phrases
            for other_phrase in similar_phrases:
                for node_a in phrase_to_nodes[phrase]:
                    node_a.generative_links = phrase_to_nodes[other_phrase]

    def get_learned_utterances(self):
        utterances = set()
        def get_utterances(node : Node, preceding : str = ""):
            utterance = preceding + " " + node.phrase if preceding else node.phrase
            if not node.children:
                utterances.add(utterance)
            for child in node.get_children():
                get_utterances(child, utterance)
        for child in self.root.get_children():
            get_utterances(child)
        return utterances

    def get_generated_utterances(self):
        utterances = set()
        def get_utterances(node : Node, preceding : str = "", generated : bool = False):
            utterance = preceding + " " + node.phrase if preceding else node.phrase
            # Only add utterances that have passed through a generative link
            if not node.children and generated:
                utterances.add(utterance)
            for child in node.get_children():
                get_utterances(child, utterance, generated)
            # TODO: Once this condition is removed, there's the possibility of loops
            if not generated:
                for link in node.generative_links:
                    get_utterances(link, preceding, True)
        for child in self.root.get_children():
            get_utterances(child)
        return utterances