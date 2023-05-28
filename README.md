```
<p align="center">
  <img src=Dirección web de una imagen que resuma el proceso alt="Alibi Logo" width="50%">
</p>
```

This repository presents the code of the EDC (Example-Dependent Cost) Counterfactual method. Also contains the code to reproduce experiments with 6 real-world datasets and to compare its performance againts the counterfactuals of other 3 state-of-the-art benchmark methods.

# EDC_counterfactuals
A method to generate Example-Dependent-Cost Counterfactuals on Neural Networks

## Table of Contents

* [Installation and Usage](#installation-and-usage)
* [Supported Methods](#supported-methods)
  * [Proposed method](#Proposed-method)
  * [Alibi CEM](#Alibi-CEM)
  * [Alibi Proto](#Alibi-Proto)
  * [GRACE](#GRACE)
* [Citations](#citations)

## Installation and Usage
The code can be downloaded directly from this repository. Also, to run the code you must create an environment. The python version and packages variates depending on which counterfactual model you want to use. The configuration of each environment can be done with Anaconda or Miniconda. The aviable counterfactual methods are:

- Alibi CEM: from https://pypi.org/project/alibi and https://github.com/SeldonIO/alibi
- Alibi Proto: from https://pypi.org/project/alibi and https://github.com/SeldonIO/alibi
- Grace: from https://github.com/lethaiq/GRACE_KDD20
- Proposed_method

### Environment for Alibi CEM and Alibi Proto methods

- Create a new environment with conda:

  ```bash
  conda create -n env_alibi python=
  ```
  
- Activate the environment:
  ```bash
  conda activate env_alibi
  ```

- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```

- Install the packages following the corresponding requirements file:
  ```bash
  pip install -r requirements_alibi_cem_and_proto.txt
  ```

### Environment for GRACE method:

- Create a new environment with conda:

  ```bash
  conda create -n env_grace python=
  ```
  
- Activate the environment:
  ```bash
  conda activate env_grace
  ```

- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```

- Install the packages following the corresponding requirements file:
  ```bash
  pip install -r requirements_grace.txt
  ```
  
### Environment for proposed_method:

- Create a new environment with conda:

  ```bash
  conda create -n env_proposed_method python=
  ```
  
- Activate the environment:
  ```bash
  conda activate env_proposed_method
  ```

- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```

- Install the packages following the corresponding requirements file:
  ```bash
  pip install -r requirements_env_proposed_method.txt
  ```

### Usage
TODO... Explicar como se ejecutan los notebooks y como se pueden reproducir los experimentos del paper

## Supported Methods
This section summarize the supported models, i.e., the porposed_method and 3 benchmark models.

### Proposed method
TODO: ...

### Alibi CEM
TODO: ...

### Alibi Proto
TODO: ...

### GRACE
TODO: ...

## Citations
If you use the proposed_method in your research, please consider citing it.

BibTeX entry:

```
...
```
