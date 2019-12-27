# -*- coding: utf-8 -*-
class CorpusReader:
    def __init__(self):
        self.train_data = []
        self.dev_data = []
        self.test_data = []
    
    def loadFile(self, filename, isTestFile=False):
        '''
        @param: filename
        @return: [[id, sentence1, sentence2, score], ...]
        '''
        lines = []
        try:
            with open(filename,'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    line = line.lower() 
                    items = line.split('\t')
                    if not isTestFile:
                        if len(items) != 4:
                            continue
                        items[3] = items[3].replace('\n', '')
                    else:
                        if len(items) != 3:
                            continue
                    lines.append(items) 
        except:
            with open(filename,'r', encoding='ISO-8859-1') as f:
                next(f)
                for line in f:
                    line = line.lower() 
                    items = line.split('\t')
                    if not isTestFile:
                        #loading the train or dev file
                        if len(items) != 4:
                            continue
                        items[3] = items[3].replace('\n', '')
                    else: 
                        #Loading the test file
                        if len(items) != 3:
                            continue
                    lines.append(items) 
        return lines
    
        
'''     
if __name__ == '__main__':
    reader = CorpusReader()
    reader.train_data = reader.loadFile('data/train-set.txt')
    reader.dev_data = reader.loadFile('data/new-dev-set.txt')
    reader.test_data = reader.loadFile('data/test-set.txt', isTestFile=True)
'''   