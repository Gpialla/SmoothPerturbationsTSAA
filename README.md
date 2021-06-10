# SmoothPerturbationsTSAA

# Smooth Perturbations for Time Series Adversarial Attacks

This is the companion repository for our paper titled "Smooth Perturbations for Time Series Adversarial Attacks". This paper is not published yet.

## Datasets used

All our experiments were performed on the widely used time series classification benchmark [UCR/UEA archive](http://timeseriesclassification.com/index.php). We used the latest version (2018) which contains 128 datasets.

## Model used

All experiments were performed using the time series classifier InceptionTime from the paper [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939). The companion Github repository is located [here](https://github.com/hfawaz/InceptionTime).

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

You should consider changing the following lines [LINES]

### Run the experiments

To simply run all the experiments, you can type:

``python3 main.py run_all``

Be careful on a single GTX 1080 Ti it can take more than a week !



# Results



## Adversarial attacks

TABLE



## Adversarial training

TABLE



## Reference

This work is not published yet.

## Acknowledgments

This work was funded by anonymous and co-funded by anonymous, anonymous, anonymous, anonymous and anonymous.

The authors would like to thank the providers of the UCR archive as well as anonymous for providing access to the GPU cluster.