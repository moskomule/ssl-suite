# ssl-suite

A Semi-Supervised Learning suite using PyTorch.

The implementation of SSL methods are based on https://github.com/google-research/mixmatch

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

## Benchmarks (on CIFAR-10)

|Number of Labeled images | ICT | Mean Teacher | MixMatch | Pseudo Label | VAT |
--- | --- | --- | --- | --- | --- |
4,000 | 0.90 | 0.89 | 0.91 | 0.82 | 0.82|

## Citation


```bibtex
@misc{homura,
    author = {Ryuichiro Hataya},
    title = {homura},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://GitHub.com/moskomule/homura}},
}
```