# Automated Bug Classifier for the VSCode Repository

Please find below a demo of the command line tool.

## 1. Environment

Exporting an environment file across platforms
```sh
conda env export --from-history > environment.yml
```

Creating an environment
```sh
conda env create -f environment.yml
```

Updating an environment
```sh
conda env update --file environment.yml  --prune
```

## 2. Documentation

For getting started with the project, run the command:
```sh
$ python3 -m app.bug_triaging -h
```
Details of the commands or instructions can be found by running:
```sh
$ python3 -m app.bug_triaging <command> -h
```
### 2.1. Data Mining
#### 2.1.1. Issue Mining
Command to mine issues from VS Code GitHub repository:
```sh  
$ python3 -m app.bug_triaging issue_mining  
```     
#### 2.1.2. Contributor Mining
Command to mine contributors from VS Code GitHub repository:
```sh
$ python3 -m app.bug_triaging contributor_mining
```

### 2.2. Data Filtering
#### 2.2.1. Issue Filtering
Command to filter issues according to specified parameters from GitHub repository:
```sh
$ python3 -m app.bug_triaging issue_filtering
```
### 2.3. Data Preprocessing
Command to preprocess the issues for training:
```sh
$ python3 bug_triaging.py preprocessing -s <path-to-issues-for-training>
$ python3 -m app.bug_triaging preprocessing -s <path-to-issues-for-training> -L --with-lemmatizer
```
_Parameters Description_:

`-s` or `--source`: Path to the issues for training

`-L` or `--with-lemmatizer`: Use lemmatization to preprocess the issues

`-T` or `--training`: Flag to enable filtering with thresholds for the training set

### 2.4 Model Training
Command to train the Random Forest Model:
```sh
$ python3 -m app.bug_triaging training
```
```sh
$ python3 -m app.bug_triaging training -a <path-to-training-set> -e <path-to-test-set> -f <path-to-feature-vectors> -r -of -om -nf -bg
```

_Parameters Description_:

`-r ` or `--recent` : Enables training with recent issues (130000 <= id <= 150000)

`-of` or `--output-feature` : Enables output of feature vectors.

`-om` or `--output-model` : Enables output of Random Forest model

`-nf` or `--no-feature-source` : Disable the usage of the feature vectors file

`-bg` or `--bigram` : Enable bigrams

### 2.5 Prediction
Command to input the id of an open vscode's issue on GitHub and provide as output a
ranked list of candidate assignees.

```sh
$  python3 -m app.bug_triaging predicting -r -id 186776
```
_Parameters Description_:

`-r` or ` --recent`: Enable predicting with the model trained with recent issues (130000 <= id <= 150000)

`-id` or `--commit-id`: The id of an open vscode's issue on GitHub.

`-v` or  `--vectorizer-source`: Path to the vectorizer file, default is data/vectorizer.joblib.

`-f <path-to-feature-vectors>` or `--source <path-to-feature-vectors>`: Path to the feature vector file, default is `data/feature_vectors.csv`

## Demo

https://github.com/deepansha16/automated-bug-classifier/assets/49023785/21c4993b-15cc-4275-8e7e-42be869756c5


