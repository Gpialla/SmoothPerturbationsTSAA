# Smooth Perturbations for Time Series Adversarial Attacks

This is the companion repository for our paper titled "Smooth Perturbations for Time Series Adversarial Attacks". 
This paper is not published yet.

## Datasets used

All our experiments were performed on the widely used time series classification benchmark [UCR/UEA archive](http://timeseriesclassification.com/index.php). 
We used the latest version (2018) which contains 128 datasets. The UCR archive is necesary to run the code.

## Model used

All experiments were performed using the time series classifier InceptionTime from the paper [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939). 
The companion Github repository is located [here](https://github.com/hfawaz/InceptionTime). InceptionTime's weight are necesary to run the code.

## Requirements

The code runs using Python 3.7. You will need to install the packages present in the [requirements.txt](requirements.txt) file.

``pip install -r requirements.txt``

## Code

The code is divided as follows:

* The [main.py](main.py) python file contains the necessary code to run the experiments.
* The [utils](utils/) folder contains the necessary functions to read the datasets and visualize the plots.

* The [attacks](attacks/) folder contains the implementation of the adversarial attack used.
* The [classifier.py](classifier.py) file contains InceptionTime implementation.

### Adaptions required

You should consider changing the [root_dir](main.py#L44) variable.
This variable specifies the folder where is the UEA/UCR archive and all the weights of InceptionTime. Both are necesary to run the code.

### Run the experiments

To simply run all the experiments, you can type:

``python3 main.py run_all``

It will first performs adversarial attacks for BIM, GM with and without noise cliping, and SGM over all the UCR archive.
Then it will produce all the pairwise plots in order to compare the methods according to the ASR and L2 norm. These plots are shown in the paper.
Finally, it will perform an adversarial attack for SGM.

Be careful, the computations can take more than a week on a single GTX 1080 Ti.

# Results



## Adversarial attacks

[This files](results/adv_attacks_results.csv) presents the adversarial attacks for all the datasets of the UCR archive.
For each method, we recorded the execution time in seconds, the ASR, the number of succesfully perturbed samples and the average L2 norm of the dataset.


## Adversarial training

The adversarial training was done for SGM over 13 datasets.

* [This file](results/adv_train_ASR.csv) presents the ASR of InceptionTime for each dataset, with and without adversarial training.
* [This file](results/adv_train_accuracy.csv) presents the accuracy of InceptionTime for each dataset, with and without adversarial training.


## Reference

This work is not published yet.

## Acknowledgments

This work was funded by anonymous and co-funded by anonymous, anonymous, anonymous, anonymous and anonymous.

The authors would like to thank the providers of the UCR archive as well as anonymous for providing access to the GPU cluster.
