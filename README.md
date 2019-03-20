# parser

This is the second asignment for the course Algorithms for speech and natural language processing of the 2018/2019 MVA Master class.

## Getting Started

### Prerequisites

- Python 3
- nltk, sklear, PYEVALB packages.

### Installation

```
pip install nltk
pip install scikit-learn
pip install PYEVALB
```

## Arguments

```--data_train```    : - File containing training data (default='sequoia-corpus+fct.mrg_strict')

```--data_test```     : - File containing test data (default='test_data')

```--train_eval```    : - Whether to split the data into train-eval datasets or train on the whole training set.

```--train_size```    : - Ratio of the data to train on, used only with train_eval option (default=0.9)

```--output_name```   : - Name of the output parse file (default='output_parse')

```--n_words_formal```: - Number of closest words w.r.t formal similarity (default=2)

```--n_words```       : - Number of closest words w.r.t embedding similarity (default=20)

```--swap```          : - Use Damerau-Levenstein distance when true and Levenstein distance otherwise.

```--lambd```         : - Float in \[0, 1\], the interpolation parameter between bigram and unigram models (default=0.8)

## Usage

To use sequoia treebank dataset to train on the first 90% and evaluate on the last 10% use the following command:
```
python main.py --train_eval --train_size 0.9
``` 
Or
```
bash run.sh --train_eval --train_size 0.9
``` 
With the default parameters you will get "output_parse" as the output file name.

To train on the whole sequoia treebank dataset and test on an input file with space-tokenized sentences, use:
```
python main.py --data_test 'test_data'
``` 
Or
```
bash run.sh --data_test 'test_data'
``` 
With the default parameters you will get "output_parse" as the output file name. If you don't specify ```--data_test``` it will by default test on 'test_data' file.

## Result

You can find the parse result of the last 10% of sequoia treebansk dataset in the file 'evaluation_data.parser_output'










