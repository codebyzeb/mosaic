"""
Opens a CHILDES corpus from a CSV file and extracts the child and child-directed utterances

"""

import sys
import pandas as pd
from pathlib import Path

from aochildes.helpers import Transcript, col2dtype, punctuation_dict
from aochildes.dataset import ChildesDataSet
from aochildes.params import ChildesParams
from aochildes.pipeline import Pipeline

from mosaic import configs

csv_path = sys.argv[1]

MIN_UTTERANCE_LENGTH = 1

df = pd.read_csv(csv_path, index_col='id', usecols=col2dtype.keys(), dtype=col2dtype)
# drop rows
print('Utterances before dropping rows: {:>8,}'.format(len(df)))
df.drop(df[df['num_tokens'] < MIN_UTTERANCE_LENGTH].index, inplace=True)
print('Utterances after  dropping rows: {:>8,}'.format(len(df)))

# Using the pre-processing from aochildes to extract child and adult utterances
child_pipeline = Pipeline()
child_pipeline.df = df.copy(deep=True)
child_pipeline.df.drop(df[~df['speaker_role'].isin(['Target_Child', 'Child'])].index, inplace=True)
child_data = ChildesDataSet()
child_data.pipeline = child_pipeline
child_data.transcripts = child_data.pipeline.load_age_ordered_transcripts()
child_utterances = child_data.load_sentences()
print("Number of child utterances: {:>8,}".format(len(child_utterances)))

adult_pipeline = Pipeline()
adult_pipeline.df = df.copy(deep=True)
adult_pipeline.df.drop(df[df['speaker_role'].isin(['Target_Child', 'Child'])].index, inplace=True)
adult_data = ChildesDataSet()
adult_data.pipeline = adult_pipeline
adult_data.transcripts = adult_data.pipeline.load_age_ordered_transcripts()
adult_utterances = adult_data.load_sentences()
print("Number of adult utterances: {:>8,}".format(len(adult_utterances)))

def write_sentences(path, sentences):
    if path.exists():
        path.unlink()

    with path.open('w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

out_path = ''.join(csv_path.split('.csv'))
write_sentences(Path(out_path + '_child.txt'), child_utterances)
write_sentences(Path(out_path + '_adult.txt'), adult_utterances)
