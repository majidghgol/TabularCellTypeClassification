# Using Trained models for Cell Type Prediction
The code in this folder is meant to deploy trained cell embeddings, feature encodings, and cell classification models for predicting cell labels on spreadsheets.

# Requirenments
running the code requires following files and libraries:

InferSent source code: https://github.com/facebookresearch/InferSent

InferSent pre-trained model: https://dl.fbaipublicfiles.com/infersent/infersent1.pkl

GloVe word embeddings: https://nlp.stanford.edu/projects/glove/

Pre-trained CE, FE, and CL models: https://drive.google.com/drive/folders/1Xs_S8kKqAsS6N_-JNbgHZnpJvLhuBHKR?usp=sharing

# How to run
Requirements need to be downloaded and their path should be passed to `predict_labels.py`. create a directory named `sample_models` in this folder (same folder as the code) and place the pre-trained CE, FE, and CL models under it.
The code gets the path to a `.xls` file and generates a json output file for predicted labels and the prediction probabilities for each sheet in the excel file.
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

# Output format
```
{
    "SHEETNAME": {
        "text": [
            ["cell11", "cell12", ...],
            ["cell21", "cell22", ...],
            ...
                ],
        "labels": [
            [l11, l12, ...],
            [l21, l22, ...],
            ...
                ],
        "label_probs": [
            [lp11, lp12, ...],
            [lp21, lp22, ...],
            ...
                ]
    }
}
```

# Notebook for visualizing predictions
`test_prediction.ipynb` contains an example of predicting cell labels and visualizing them.

