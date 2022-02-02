class DataSet:
    
    def __init__(self, filename : str):
        self.utterances = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                self.utterances.append(line.strip().split(' '))
        self.size = len(self.utterances)
