"""
This script writes line-by-line text files to disk.

"""

from aochildes.dataset import ChildesDataSet
from aochildes.params import ChildesParams
from aochildes.pipeline import Pipeline

from mosaic import configs

MAX_AGE = 365*6
CORPUS_NAME = "aochildes_under6"

def write_sentences(path, sentences):
    if path.exists():
        path.unlink()

    with path.open('w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

# First get all utterances in AO-CHILDES to extract speaker roles
all_utterances = Pipeline(ChildesParams(bad_speaker_roles=[]))
speakers = all_utterances.df.groupby("speaker_role").size()

# Then extract child and adult utterances and write them
child_utterances = ChildesDataSet(ChildesParams(max_days=MAX_AGE, bad_speaker_roles=list(speakers.keys().drop("Child")))).load_sentences() # drop all but child utterances
adult_utterances = ChildesDataSet(ChildesParams(max_days=MAX_AGE)).load_sentences() # default parameter is to drop child utterances

print(f'Loaded {len(child_utterances):,} child utterances and {len(adult_utterances):,} adult utterances')

write_sentences(configs.Dirs.corpora / f'{CORPUS_NAME}_child.txt', child_utterances)
write_sentences(configs.Dirs.corpora / f'{CORPUS_NAME}_adult.txt', adult_utterances)
