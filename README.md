# VRDL-Homework1

## Requirements
The following package is used in this homework
```
numpy==1.19.3
torch==1.9.0+cu111
torchvision==0.10.0+cu111
```
## Folder structure

    .
    ├──dataset
       ├──Testing
          ├──XXXX.jpg
       ├──Training
          ├──XXXX.jpg
       ├──classes.txt           # contain 200 bird species and it's number (e.g. 001.Black_footed_Albatross)
       ├──training_labels.txt   # filename and label mapping (e.g. 4283.jpg 115.Brewer_Sparrow)
       ├──testing.txt           # test filename (e.g. 4282.jpg)
       ├──fold{i}.txt           # we split training data into 5 fold (i=1,2,3,4,5)
    ├──functions                # functions inside
    ├──saved_model              # trained model weights
    ├──split_folds.py           # split training dataset into 5 folds (fold{i}.txt)
    ├──train.py
    ├──inference.py
    └──README.md


## Training code
If you want to change the hyper-parameter, modify a few lines (line 18 to 29) of content in `train.py` <br />
Note: `FoldID = i` indicates that the training dataset in `fold{i}.txt` will be the validation dataset <br />
To run this code, use the following command

```
python train.py
```

## Pre-trained model
After running `train.py`, the model weight with the highest accuracy in that FoldID will be saved in saved_model <br />
The format of the file name is `fold=X_ep=YY_acc=0.ZZZZ.pth`


## Inference code
`inference.py` will catch the model weight in saved_model automatically. To reproduce the testing result, run
```
python inference.py
```


