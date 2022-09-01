# FastText-Classification-on-SLED
 
Training a fastText model on the SLED categorization dataset.

## Main conclusions

The fastText model, trained on Slovene embeddings achieved slightly (2 points) better results than the model that was not trained on the embeddings. The highest micro and macro F1 scores that were achieved on this task are 0.85. Training on the trainlarge gives only slightly better results (2 points) for the model that was not trained with embeddings, while it takes much more time than with trainsmall (53 minutes versus 14 minutes for 800 epochs). For the model, trained on the embeddings, there is no difference between the trainsmall and trainlarge.

| Embeddings, train file | Micro F1 | Macro F1 |
|:---------:|---------:|----------|
|    yes, trainsmall      |  0.85     |  0.85   |
|    yes, trainlarge      |  0.85       |   0.85      |
|    no, trainlarge      |   0.85       |   0.85      |
|    no, trainsmall      |    0.83      |    0.83      |


The hyperparameter search, focused on the number of epochs, revealed optimum numbers to be quite high - 800 epochs for the model without the embeddings, and 400 epochs for the model with the embeddings. Other hyperparameters were set to default values. When training on the trainlarge, the optimal number of epochs was even bigger: 900 for the model without the embeddings, 1000 for the model with embeddings.


## FastText model without the embeddings (trainsmall)

During the hyperparameter search, I only experimented with different numbers of epochs. I did not use the automatic hyperparameter search, but did the experiments "manually", by training the models on the train split, with different numbers of epochs each time, and evaluating them on the dev split.

![](results/hyperparameter-search-epoch-number.png)

As we can see from the plot, the micro and macro F1 scores keep rising until the epoch 800, afterwards, the scores remain around 0.83. For testing, I used 800 epochs, other hyperparameters were left on default values.

| Tested on | Micro F1 | Macro F1 |
|:---------:|---------:|----------|
|    dev    |   0.8346       |   0.8348       |
|    test      |    0.8285      |    0.8282      |

Classification report:

![](results/classification-report.png)

Confusion matrix for test file:

![](results/confusion-matrix-on-test.png)


## FastText model with the embeddings (trainsmall)

I used Slovene embeddings from the CLARIN.SI repository: Word embeddings CLARIN.SI-embed.sl 1.0 (https://www.clarin.si/repository/xmlui/handle/11356/1204).

The hyperparameter search was conducted in the same manner as in the previous experiment.

![](results/hyperparameter-search-with-embeddings-epoch-number.png)

As we can see from the plot, the micro and macro F1 scores keep rising until the epoch 400, afterwards, the scores remain around 0.85 which is not much higher than the results of the model without the embeddings, evaluated on dev. For testing, I used 400 epochs, other hyperparameters were left on default values.


| Tested on | Micro F1 | Macro F1 |
|:---------:|---------:|----------|
|    dev    |   0.8461       |   0.8457       |
|    test      |  0.8492     |  0.8487  |

Classification report:

![](results/classification-report-with-embeddings.png)

Confusion matrix for the test file:

![](results/confusion-matrix-on-test-with-embeddings.png)


## FastText model without the embeddings (trainlarge)

### Hyperparameter search

Similarly to the model, trained on trainsmall train split, the optimum number of epochs revealed to be quite large - the scores stop rising at 900 epochs -> I used 900 epochs as the optimum value.

![](results/hyperparameter-search-epoch-number-trainlarge.png)

| Tested on | Micro F1 | Macro F1 |
|:---------:|---------:|----------|
|    dev    |   0.860       |   0.861       |
|    test      |   0.8515       |   0.8533      |

Classification report:

![](results/classification-report-trainlarge.png)

Confusion matrix:

![](results/confusion-matrix-on-test-trainlarge.png)

## FastText model with embeddings (trainlarge)

While the optimal number of epochs for trainsmall with embeddings was 400 epochs, the hyperparameter showed that when training on trainlarge, the optimal number is much higher - even after 1000, the scores kept rising (although slowly). As training on 1000 epochs takes more than 100 minutes, I stoped searching for the optimum epoch number after 1000 epochs and used this number for testing.

![](results/hyperparameter-search-trainlarge-with-embeddings-epoch-number.png)

| Tested on | Micro F1 | Macro F1 |
|:---------:|---------:|----------|
|    dev    |   0.860       |   0.8603       |
|    test      |   0.8531       |   0.8537      |

Classification report:

![](results/classification-report-with-embeddings-trainlarge.png)

Confusion matrix:

![](results/confusion-matrix-on-test-trainlarge-with-embeddings.png)
