# Pre-trained Cell Embeddings for Cell Classification

## pre-requisits 
### python libraries
The code is in Python 3, and tested in Conda environment. First create a fresh conda environment:

```
conda create -n cell_emb python=3.7
```
After successfully creating the environment, activate it using the `source activate cell_emb` command. Then install the required packages using the `requirements.txt` file:

```
pip install -r requirements.txt
```

### GLOVE word vectors
Download and unzip GLoVe embeddings:

```
cd deploy
mkdir GloVe
curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip GloVe/glove.840B.300d.zip -d GloVe/
```

### Infersent source code
```
git clone https://github.com/facebookresearch/InferSent
```

### Infersent pre-trained model
```
curl -Lo InferSent/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
```

## Input format
The code expects the tables to be in json line format. Each json object represents a single table. For example a spreadsheet in an excel file, will be a single table. The table objects are identified by the file path (and the sheet name in case of excel file), and shoul contains the following fields:

```
- annotations: 2D list of cell labels, or None
- directory: path to the file (excel/csv)
- feature_array: 3D list of cell stylistic features
- feature_names: list of feature names (in the same order appearing in feature_array)
- file_name: file name (excel/csv)
- num_cols: number of table cols
- num_rows: number of table rows
- table_array: 2D list of cell content strings
- table_id: sheet name (excel) / Xpath (html) / None (csv)
- url: url to the page with table (html)
```

## Supported formats
Currently, excel files are supported and tools for converting excel files to this json line format are available. The libraries are in `code/src/excel_toolkit.py`. __TODO__: put a script, given a folder produes a dataset for training.

## Using pre-trained models
We have pre-trained some models using both stylistic features, and cell embeddings, trained on our `cius` dataset. These models are available in the Google Drive link below:

https://drive.google.com/drive/folders/1Xs_S8kKqAsS6N_-JNbgHZnpJvLhuBHKR?usp=sharing

You can use `deploy/predict_labels.py` python script for generating predictions for a given excel sheet. The script usage is as follows:

```
usage: predict_labels.py [-h] [--file FILE] [--ce_model CE_MODEL]
                         [--fe_model FE_MODEL] [--cl_model CL_MODEL]
                         [--w2v W2V] [--vocab_size VOCAB_SIZE]
                         [--infersent_model INFERSENT_MODEL]
                         [--infersent_source INFERSENT_SOURCE] [--out OUT]

processing inputs.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           path to the .xls spreadsheet.
  --ce_model CE_MODEL   path to the trained cell embedding model.
  --fe_model FE_MODEL   path to the trained feature encoding model.
  --cl_model CL_MODEL   path to the trained classification model.
  --w2v W2V             path to the glove embeddings.
  --vocab_size VOCAB_SIZE
                        w2v vocab size.
  --infersent_model INFERSENT_MODEL
                        path to the infersent model.
  --infersent_source INFERSENT_SOURCE
                        path to the infersent source code.
  --out OUT             path to the output json file.

```

# Training your own models
To train your own models, you should first create a `.jl.gz` file containing the tables in your dataset. For training the _cell_embeddings_ and _feature_encoding_ models you do not need annotations and can use unlabeled tables. To train _classification_model_ however, you need a dataset where `annotations` field of the table json objects are populated. We explain how you can annotate your own tables and use the tools in this repo to create such dataset.

## Train cell embedding model


## Train stylistic features encoding model

## Train cell classification model

## Reference and citation
Please cite our ICDM'19 paper.
```
@inproceedings{gol2019tabular,
  title={Tabular Cell Classification Using Pre-Trained Cell Embeddings},
  author={Gol, Majid Ghasemi and Pujara, Jay and Szekely, Pedro},
  booktitle={2019 IEEE International Conference on Data Mining (ICDM)},
  pages={230--239},
  year={2019},
  organization={IEEE}
}
```