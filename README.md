# **EDC Counterfactuals Repository**

This repository presents the code of COCOA: Cost-Optimized COunterfactuAl explanation method. Also contains the code to reproduce experiments with 6 real-world datasets and to compare its performance against the counterfactuals of other 3 state-of-the-art benchmark methods.

>The COCOA method generates Example-Dependent-Cost Counterfactuals on Neural Networks. The differences between a traditional class counterfactual and a cost-optimised counterfactual could be represented with the following graph:

<p align="center">
  <image src="images/counterfactual_boundaries.png" alt="" style="width: 500px;" >
</p>

<p style="text-align: center;">
Counterfactual examples $x^c$ for a given pattern x (the factual) in a conventional problem and in an EDC problem, where the decision boundaries are different.
</p>


# **Table of Contents**

* [1. Installation and Usage](#installation-and-usage)
* [2. Supported Methods](#supported-methods)
  * [2.1 COCOA](#COCOA-method)
  * [2.2 Alibi CEM](#Alibi-CEM)
  * [2.3 Alibi Proto](#Alibi-Proto)
  * [2.3 GRACE](#GRACE)
  * [3. Paper Citation](#Citation)

# **1. Installation and Usage**
The code can be downloaded directly from this repository. Also, to run the code you must create an environment for each counterfactual method. The Python version and packages variate depending on which counterfactual model you want to use. The configuration of each environment can be done with Anaconda or Miniconda. The available counterfactual methods are:

- COCOA_method: The novel method
- Alibi CEM: from https://pypi.org/project/alibi and https://github.com/SeldonIO/alibi
- Alibi Proto: from https://pypi.org/project/alibi and https://github.com/SeldonIO/alibi
- Grace: from https://github.com/lethaiq/GRACE_KDD20

Each of these counterfactual methods has a corresponding notebook to reproduce the tables of results of the whole paper, and also to reproduce the experiments of each counterfactual method. These notebooks must be executed with the specified environment depending on the method as explained in the next sections. 

The notebooks are:
- COCOA_method: Compute_experiments_with_COCOA.ipynb ; execute with [env_cocoa](#Environment-for-COCOA_method) environment
- Alibi CEM: Compute_experiments_with_Alibi_CEM.ipynb ; execute with [env_alibi](#Environment-for-Alibi-CEM-and-Alibi-Proto-methods) environment
- Alibi Proto: Compute_experiments_with_Alibi_Proto.ipynb ; execute with [env_alibi](#Environment-for-Alibi-CEM-and-Alibi-Proto-methods) environment
- Grace: Compute_experiments_with_GRACE.ipynb ; execute with [env_grace](#Environment-for-GRACE-method) environment

## 1.1. Environment for COCOA_method:

- Create a new environment with conda:

  ```bash
  conda create -n env_cocoa python=3.8.16
  ```
  
- Activate the environment:
  ```bash
  conda activate env_cocoa
  ```

- Upgrade pip:
  ```bash
  pip install --upgrade pip
  ```

- Install the packages following the corresponding requirements file:
  ```bash
  pip install -r requirements_COCOA.txt
  ```

## 1.2. Environment for Alibi CEM and Alibi Proto methods

- Create a new environment with conda:

  ```bash
  conda create -n env_alibi python=3.8.13
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
  pip install -r requirements_alibi_CEM_and_Prototypes.txt
  ```

## 1.3. Environment for GRACE method:

- Create a new environment with conda:

  ```bash
  conda create -n env_grace python=3.8.16
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
  pip install -r requirements_GRACE.txt
  ```

## 1.4. Usage
Each notebook is organised with the same structure:

- *Section 1. Libraries*
    > Install the dependences


- *Section 2. Load params from yaml*
    > Load the necessary parameters for the notebook

- *Section 3. Compute the metrics and results table from pre-searched counterfactuals*
    > Recompute the results tables of the [Paper Citation](#Citations) using the precomputed counterfactuals (stored in the pre-computed_counterfactual_explanations folder) for all the tested methods and datasets.

- *Section 4. (Optional) Reproduce the counterfactual searching over all datasets*
    > Recompute the counterfactuals for the specified datasets with the notebook explanation method and stores it in the re-computing_counterfactual_explanations folder. To visualise these results, the user must re-execute Section 3 but modifying the **"use_precomputed_counterfactuals"** parameter. Some datasets and counterfactual methods need high computational resources and time to compute all the counterfactuals, therefore, by default all the notebooks have the **small_test** parameter set to True which limits the experiment just to 2 counterfactuals for the dataset. If full experimentation is required by the user, this parameter must be set to False.


# **2. Supported Methods**
This section summarizes the supported models, i.e., the proposed COCOA method and 3 benchmark models. A more detailed explanation of each counterfactual method is provided in the [Paper Citation](#Citations) and the corresponding original paper of each method.

## 2.1. COCOA method
The novel method

The objective of the proposed COCOA method is to obtain the counterfactual example that is at the minimum distance and guarantees a positive impact on the decision costs.

[Paper Citation](#Citations)

<p align="center">
  <image src="images/counterfactual_boundaries.png" alt="" style="width: 500px;" >
</p>

<p style="text-align: center;">
Counterfactual examples $x^c$ for a given pattern x (the factual) in a conventional problem and in an EDC problem, where the decision boundaries are different.
</p>

## 2.2. Alibi CEM
The benchmark 1.

The CEM method (Contrastive Explanation Method) searches for counterfactual examples that explain the decisions of black-box models.

```
@inproceedings{Dhurandhar2018,
               title={Explanations based on the missing: Towards contrastive explanations with pertinent negatives},
               author={Dhurandhar, Amit and Chen, Pin-Yu and Luss, Ronny and Tu, Chun-Chen and Ting, Paishun and Shanmugam, Karthikeyan and Das, Payel},
               booktitle={Advances in Neural Information Processing Systems},
               pages={592--603},
               year={2018}
}
```

## 2.3. Alibi Proto
The benchmark 2.

The Prototypes method seeks to obtain counterfactual samples at a minimum distance guiding the process with prototypes.

```
@inproceedings{van2021interpretable,
               title={Interpretable counterfactual explanations guided by prototypes},
               author={Van Looveren, Arnaud and Klaise, Janis},
               booktitle={Machine Learning and Knowledge Discovery in Databases. Research Track: European Conference,
               ECML PKDD 2021, Bilbao, Spain, September 13--17, 2021, Proceedings, Part II 21},
               pages={650--665},
               year={2021},
               organization={Springer}
}
```

## 2.4. GRACE
The benchmark 3.

The GRACE method (GeneRAting Contrastive samplEs) is focused on Neural Network models and aims to find counterfactual examples at a minimum distance that produce a switch in the classification label.

```
@article{le2019grace,
         title={GRACE: Generating Concise and Informative Contrastive Sample to Explain Neural Network Model's Prediction},
         author={Thai Le and Suhang Wang and Dongwon Lee},
         year={2019},
         journal={Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '20)},
         doi={10.1145/3394486.3403066}
         isbn={978-1-4503-7998-4/20/08}
         }
```

# **3. Paper Citation**
If you use the COCOA method in your research, please consider citing it.

BibTeX entry:
```
@article{mediavilla2024cocoa,
         title={COCOA: Cost-Optimized COunterfactuAl explanation method},
         author={Mediavilla-Rela{\~n}o, Javier and L{\'a}zaro, Marcelino},
         journal={Information Sciences},
         volume={670},
         pages={120616},
         year={2024},
         publisher={Elsevier}
         }
```

