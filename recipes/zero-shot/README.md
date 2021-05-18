# Improving Zero-Shot Translation by Disentangling Positional Information

## Software
### Dependency
For training:
* Python version >= 3.7 (most recommended)
* [PyTorch](https://pytorch.org/) >= 1.0.1
* [apex](https://github.com/nvidia/apex) when using half- or mixed-precision training 
  
For preprocessing:
* [subword-nmt](https://github.com/rsennrich/subword-nmt)
* [sentencepiece](https://github.com/google/sentencepiece)
* [indic_nlp_library](https://github.com/anoopkunchukuttan/indic_nlp_library) and [FloRes](https://github.com/facebookresearch/flores) for Indian language tokenization
* [Moses](https://github.com/moses-smt/mosesdecoder) for tokenization of languages in Latin script

For evaluation:
* [sacreBLEU](https://github.com/mjpost/sacrebleu)

### Preprocess
First source the config file:
```
source ./recipes/zero-shot/config.sh
```
Preprocess and binarize the dataset:
```
bash ./recipes/zero-shot/prepro.sh
bash ./recipes/zero-shot/binarize.sh
```

### Train
To train the baseline:
```
bash ./recipes/zero-shot/train.sh
```

To train the baseline with residual removed in the middle encoder layer:
```
bash ./recipes/zero-shot/train.remove.residual.sh
```

To train the baseline with residual removed and use position-based query, in the middle encoder layer:
```
bash ./recipes/zero-shot/train.remove.residual.query.sh
```

### Test
```
bash ./recipes/zero-shot/pred.sh $PATH
```
Here $PATH contains the trained model.

## Dataset
### IWSLT

The training set is a subset of the TED task from IWSLT 2017.
Provided from the MMCR4NLP corpus described in https://arxiv.org/pdf/1710.01025.pdf.

We use the dev set from multilingual TED task from IWSLT 2017 (dev2010).
https://wit3.fbk.eu/2017-01

The test set is the official test set (tst2017) for the multilingual TED task from IWSLT 2017.
https://wit3.fbk.eu/2017-01-b

### Europarl

The training, dev, and test set come from a subset of the TED task from Europarl.
Provided from the MMCR4NLP corpus described in https://arxiv.org/pdf/1710.01025.pdf.

### PMIndia

As the corpus (http://data.statmt.org/pmindia/) did not specify paritions of the train / dev / test sets, 
we partition the corpus ourselves.
After deduplicating, we first take a multiway subset of all languages, using English as pivot. 
This results in 1695 sentences in the dev and test respectively.
In this process, we excluded Assamese  (as),  Maithili  (mni),  Urdu  (ur). 
As these languages had very little data, including them would make the multiway subset too small. 
We upload the dev and test sets [here]().


## Experiment and Results
### Computation Infrastructure
All models were trained on Nvidia GeForce RTX 2080ti GPUs. 

### Average Runtime
All experiments were run with mixed-precision (optimization level "O1" from [apex](https://github.com/NVIDIA/apex)).

We train for 64 epochs by default.
The one exception is the Europarl-full case, 
where we also include the zero-shot directions in dev set for early stopping.

| Dataset                           | # sentences | Layers | Average Time Per Epoch |
|-----------------------------------|--------|--------|-----------------------| 
| IWSLT                             | 870K | 5 | 7 min   |
| Europarl (non-overlap / multiday) | 2M   | 8 | 45 min  |
| Europarl (full)                   | 17M  | 8 | 7 hr    |
| PMIndia                           | 639K | 5 | 9 min   |

### Number of Parameters
Note that although the models for IWSLT and PMIndia have the same depth, the vocabulary sizes differ. 
The final parameter count is therefore also different.

| Dataset   | Layers | # Parameters      |
|-----------|--------| ------------------|
| IWSLT     |  5     | 47M               |
| Europarl  |  8     | 79M               |
| PMIndia   |  5     | 58M               |

### Validation and Test Performance

The average BLEU scores on the supervised directions are reported below 
in the format of: dev / test.

| Dataset   | Baseline | Residual | Residual + Query |
|-----------|----------|----------|---------|
| IWSLT                  | 27.6 / 29.8 | 27.2 / 29.4 | 27.4 / 29.4 |
| Europarl, multiway     | 34.6 / 34.2 | 34.2 / 33.9 | 34.8 / 33.1 |
| Europarl, non-overlap  | 36.0 / 35.6 | 35.7 / 35.4 | 35.3 / 34.9 |
| Europarl, full         | 35.9 / 35.4 | 35.6 / 36.4 | 36.0 / 35.9 |
| PMIndia                | 30.6 / 30.4 | 30.0 / 29.9 | 29.4 / 29.2 |
