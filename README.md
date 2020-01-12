# ssl-suite

A Semi-Supervised Learning suite using PyTorch.

The implementation of SSL methods are based on https://github.com/google-research/mixmatch

Currently, the following methods are implemented:
* Interpolation Consistency Training
* Mean Teacher
* MixMatch
* Pseudo Label
* Virtual Adversarial Training

## Updates

* 2019/12/25 Update WideResNet compatible with google's implementations

## Requirements

* Python>=3.7
* PyTorch>=1.3
* torchvision>=0.4.2
* homura>=2019.11 (`pip install -U git+https://github.com/moskomule/homura@v2019.11`)
* hydra>=0.11 (`pip install -U hydra-core`)

For data preparation, run `backends/data.py`.

## How to run

`python {ict,mean_teacher,mixmatch,pseudo_label,vat}.py`

If you want to change configurations from the default values, do something like

`python METHOD.py data.name=cifar100`

For configurable values, see files in `config`.

## Benchmarks

Following Berthelot+2019, the reported accuracy values are median of accuracy of last 20 epochs.

### CIFAR-10

|Number of Labeled images | ICT | Mean Teacher | MixMatch | Pseudo Label | VAT |
--- | --- | --- | --- | --- | --- |
4,000 | 0.89 | 0.89 | 0.93 | - | - |

* Supervised learning on 50,000/4,000 images yields accuracy of 0.94/0.82.

### SVHN

`python {ict,mean_teacher,mixmatch,pseudo_label,vat}.py data.name=svhn data.labeled_size=1000 data.unlabeled_size=64931 data.val_size=7326 ${MODEL_SPECIFIC_SETTINGS}`

|Number of Labeled images | ICT | Mean Teacher | MixMatch | Pseudo Label | VAT |
--- | --- | --- | --- | --- | --- |
1,000 | 0.91 | 0.96 | 0.94 | - | - |

* Supervised learning on 50,000/4,000 images yields accuracy of 0.97/0.88.
 
## Citation


```bibtex
@misc{ssl-suite,
    author = {Ryuichiro Hataya},
    title = {ssl-suite: Semi-supervised Learning suite using PyTorch},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/moskomule/ssl-suite}},
}
```