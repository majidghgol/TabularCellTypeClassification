# Example: training models using SAUS dataset

To follow the steps explained in this example, please start from the root directory of the repository in the terminal.

## Install pre-requisites
Please make sure you have the pre-requisites set up. Please refer to the main README of the repository for how to set them up.

## Create a directory to store models
```
mkdir models
mkdir models/fine_tuned
```

## Update `specs_saus.json` with your own paths
the paths to InferSent model, and GloVe embeddings should be updated:

```
"w2v_path": "PATH TO glove.840B.300d.txt",
"infersent_model": "PATH TO infersent1.pkl",
```


## Train cell embedding model
```
python code/src/train_cl.py --infersent_source PATH_TO_INFERSENT --spec_path ./example/specs_saus.json
```

## Train feature auto-encoder model
```
python code/src/train_fe.py --infersent_source PATH_TO_INFERSENT --spec_path ./example/specs_saus.json
```

## Train cell classification model
```
python code/src/train_cl.py --infersent_source PATH_TO_INFERSENT --spec_path ./example/specs_saus.json
```