from string import punctuation
class DataSet:
    
    def __init__(self, filename : str):
        self.utterances = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                self.utterances.append(line.strip().split(' '))
        self.size = len(self.utterances)

    def strip_punctuation(self):
        for utterance in self.utterances:
            for punct in punctuation:
                if punct in utterance:
                    utterance.remove(punct)
    