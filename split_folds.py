import os
import random
from collections import defaultdict

def construct_fold(loc, Mapping):
    '''
    For each key (label), pop 3 items from the value (namelist)
    write them into txt file, the format is the same as training_labels.txt
    '''
    fp = open(loc, 'w')
    for label, namelist in Mapping.items():
        for _ in range(3):
            name = namelist.pop()
            fp.write(name+' '+label+'\n')
    fp.close()

if __name__ == '__main__':
    # read informations form training_labels.txt, we will got something like 
    # [['4283.jpg, '115.Brewer_Sparrow'], ['3982.jpg', '162.Canada_Warbler'],..]
    TrainLabel = [line.strip().split(' ') 
             for line in open(os.path.join('dataset', 'training_labels.txt'))]

    # Construct a dictionary that maps label to list of filenames
    # we will got something like
    # {'115.Brewer_Sparrow':['4283.jpg', '0588.jpg' ...], ...}
    Mapping = defaultdict(list)
    for name, label in TrainLabel:
        Mapping[label].append(name)

    # From here, we can see each label has 15 training data
    '''
    for k,v in Mapping.items():
        print(len(v), k)
    '''

    # So, we pick 3 training data from each label to form a fold
    # Shuffle
    random.seed(0)
    for label, namelist in Mapping.items():
        random.shuffle(namelist)

    # Write txt
    for fold_id in range(1,6):
        construct_fold(loc=os.path.join('dataset', 'fold'+str(fold_id)+'.txt'),
                       Mapping = Mapping)
