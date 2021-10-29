import re
import os
import glob

def read_classes_mapping(root = 'dataset', filename = 'classes.txt'):
    # {'001':'Black_footed_Albatross', ... }
    loc = os.path.join(root, filename)
    return dict([line.strip().split('.') for line in open(loc, 'r')])

def read_folds(root = 'dataset', filelist = ['fold1.txt', 'fold2.txt']):
    pairs = []
    for filename in filelist:
        loc = os.path.join(root, filename)
        pairs += [line.strip().split(' ') for line in open(loc, 'r')]
    return zip(*pairs)
        
def read_test(root = 'dataset', filename = 'testing.txt'):
    # ['1234.jpg', '1357.jpg', ...]
    loc = os.path.join(root, filename)
    return [line.strip() for line in open(loc, 'r')]

def remove_with_prefix(root, prefix):
    for filename in glob.glob(os.path.join(root, prefix+'*')):
        os.remove(filename)
    return

def read_models(root, model_cls):
    # Assume that there some .pth file in the folder
    # filename is of the format: "fold=X_ep=XX_acc=0.XXXX.pth"
    # Try to extract the models with the highest accuracy in each fold (1-5)

    def parse(filename):
        ID, _, acc = re.findall('\d+\.*\d*', filename)
        return int(ID), float(acc)

    def compare(filename1, filename2):
        # return True if filename1 acc > filename2 acc
        if filename1 is None:
            return False
        ID1, acc1 = parse(filename1)
        ID2, acc2 = parse(filename2)
        return acc1 > acc2
    
    NameList = [None] * 5
    for filename in glob.glob(os.path.join(root, 'fold*')):
        ID, _ = parse(filename)
        if not compare(NameList[ID-1], filename):
            NameList[ID-1] = filename

    ModelList = []
    for loc in NameList:
        if filename is not None:
            ModelList.append(model_cls())
            ModelList[-1].load(loc)
            ModelList[-1].eval()

    return ModelList
        
