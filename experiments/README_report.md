# EXECUTIVE REPORT ABOUT ACCURACY OF TRAINING MODELS

## Steps to develop the training phase of the project

The first thing we did was analyze the photos without removing the background, and train models based on the train and validation set already configured. To do this, the following steps were followed, using only 25 epochs, in order to let the server free for my other fellows too:

1° I defined a basemodel using the config file with the default parameters (the ones present in the example config file).

2° I started to change some parameters, in order to try to get the best value of loss and accuracy, without focuse not much, just for now, in the overfitting.

3° Additionally to the previous step, also was reviewed if the value of loss decrease after each epoch, and the value of accuracy increase after each epoch.

4° During these first steps I decided to use the Epsilon parameter, which according to the official documentation, have a recomended value between 0.1 and 1, and this greatly improved the accuracy values, but with an increase on the GAP between training and validation accuracy. For that reason, the next steps were make our best effort to reduce overfitting.

5° To achieve that, I increased the value of the parameters for data-augmentation, and also the value of dropout, and used a combination of different values to those features, in order to get the best for our goal to reduce the GAP of overfitting.

6° Additionally to the previous step, also was tested the use of a regularization method, between "l1 and l2", and "l2", in order to analyze the best combination with the other parameters. It is important to mention that, when I used these regularizations, I increase the learning rate, in order to not punish a lot the velocity of the model training.


## Results with original photos

After some iterations using all the steps mentioned before, I get my best combination, that is in the config-file of the experiment 16 (see experiments folder). It is important to mention that, in order to keep a good value of accuracy, I prefered to keep some overfitting in the model, but with the condition that the model and weights will be keep if the evaluation on the test set give us a good accuracy.

To finish this part, our best model was retrained with the config-file of the experiment 16, but with 50 epochs, and the results on the train and validation sets were the following:

In epoch 41 of 50 / Train loss: 0.7482 - Train accuracy: 0.7953 / Validation loss: 1.5866 - Validation accuracy: 0.5921

![accuracy_exp_016](https://github.com/anyoneai/sprint5-project/blob/LUISPENALOZA_assignment/experiments/graphics/acc_016.JPG)

![loss_exp_016](https://github.com/anyoneai/sprint5-project/blob/LUISPENALOZA_assignment/experiments/graphics/acc_016.JPG)

The last step, in this part, was to evaluate our trained model on the test set (using our jupyter notebook file), to analyze if it is worth to keep this model as our best option. The results was quite good, an accuracy of 0.4953. The detailed results were the following:

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| accuracy     |           |        | 0.50     | 8041    |
| macro avg    | 0.59      | 0.49   | 0.50     | 8041    |
| weighted avg | 0.59      | 0.50   | 0.50     | 8041    |


## Results with cropped photos

Finally, I cropped the photos, and with that, retrained the model with our best config-file of the previous stage, having the results on the experiment 17 (see experiments folder), with 50 epochs, obtaining the following train and validation results:
In epoch 48 of 50 / Train loss: 0.4688 - Train accuracy: 0.8748 / Validation loss: 1.1340 - Validation accuracy: 0.7039

![accuracy_exp_017](https://github.com/anyoneai/sprint5-project/blob/LUISPENALOZA_assignment/experiments/graphics/acc_017.JPG)

![loss_exp_017](https://github.com/anyoneai/sprint5-project/blob/LUISPENALOZA_assignment/experiments/graphics/loss_017.JPG)

And, as the last previous step, I evaluated our trained model on the cropped photos of the test set, and give us a better results, with an accuracy of 0.6430. The detailed results of this last evaluation were the following:

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| accuracy     |           |        | 0.64     | 8041    |
| macro avg    | 0.71      | 0.64   | 0.64     | 8041    |
| weighted avg | 0.70      | 0.64   | 0.64     | 8041    |

You can also check the whole evaluation in the file "Final Model Evaluation" inside the notebooks folder.


## Conclusions

As we see, the best results were with the cropped photos. That is because is easier for the model to identify the car without much noise in the background, and focusing only in the important details of the image.

Also, we could notice that the epsilon recommended parameter was a really potential factor to increase the accuracy of our trained model; with the disadvantage of overfitting, that we balance with the increase of the factors in the dropout and data augmentation parameters.

Finally, it is important to mention that I decided to keep the learning rate of adam in its default factor, and without using any regularization method in the dense layer. That was because I make some experiments and combination using different values in this two parameters, but the results were not improving. It is a possibility that, if I would have continued iterate with differents combinations of this two parameters, and also the ones of epsilon, dropout, data augmentation, epochs, etc., I could have obtained better results, so there is space to increase the accuracy and dicrease the overfitting, only a matter of resources and time that we have to keep in balance depending on the context.

Let's see one last graphic seeing the results of our best trained models in the original and cropped photos

![accuracy_exp_016_017](https://github.com/anyoneai/sprint5-project/blob/LUISPENALOZA_assignment/experiments/graphics/acc_016_017.JPG)

![loss_exp_016_017](https://github.com/anyoneai/sprint5-project/blob/LUISPENALOZA_assignment/experiments/graphics/loss_016_017.JPG)


> THANKS FOR READING!

Please do not hesitate to contact me with feedback, suggestions, good practices, or more information about the experiments, results, and other requeriments you need.
