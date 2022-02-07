import math

from typing import List
from dataclasses import dataclass, field

class Params:
    
    # For Node Creation Probability calculations
    corpus_size : int = 0
    calculate_ncp : bool = True
    m : int = 20

    # For computing generative links
    generative_links_using_joint_contexts : bool = False
    overlap_threshold : float = 0.2
    hide_progress_bar : bool = True