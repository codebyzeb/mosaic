"""
Download a corpus from CHILDES and save a csv for each target child

"""

from mosaic import configs
import childespy

CORPUS = "Manchester"
COLLECTION = "Eng-UK"

utts = childespy.get_utterances(collection=COLLECTION, corpus=CORPUS)
speakers = list(utts["target_child_name"].unique())
for speaker in speakers:
    a = utts[utts["target_child_name"] == speaker]
    path = configs.Dirs.corpora / f'{CORPUS}'
    if not path.exists():
        path.mkdir()
    path = path /f'{speaker}.csv'
    if path.exists():
        path.unlink()
    a.to_csv(path)

