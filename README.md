# capstone  - kitchenware classification

## **Problem statement:**

By analyzing data from a labeled dataset of RGB images, the objective is to classify the type of kitchenware present in the pictures.

The target classes are:

* cup
* plate
* spoon
* fork
* knife
* glass

## **Dataset Origin:**

Kitchenware Classification Kaggle competition: [[competition home page](https://www.kaggle.com/competitions/kitchenware-classification/overview)] (https://www.kaggle.com/competitions/kitchenware-classification/overview)

The instructions to download the dataset are in the Data section (they are also included in the beginnning of the notebook in thid repo)

## **Dataset Information:**

The dataset consists of 5559 labeled images, that in the notebook are splitted in train (80%) and validation(20%). The test set has been hold out, being object of the Kaggle competition (see dataset origin below). The labeled images were produced through toloka.ai services.

## Description of the Project and of the steps to reproduce it

The first part of the project (data preparation and experiments for model selection and hyperparameter tuning) was executed in the cloud by means of the services offered by Saturn Cloud (), to take advantage of a performant GPU processing unit. For the sake of reproducibility, a conda environment (requirements.txt) and a train.py are available here, to reproduce this part of the project in a local environment (but without GPU aidance). For those who only want to check the model deployment, it is sufficient for them to make use of the weights of the final model stored in xception_capstone_08_0_925.h5 and of service.py and test.py to test the service in local, as explained in detail below.

* *notebook.ipynb* contains EDA and the final Model selection.
  sandbox.ipynb contains a messy :) ,constituted by various experiments that eventually converged in the notebook.ipynb,
* *train.py* contains the logic for training from CLI the final model determined in notebook.ipynb

for the deployment of the model as a service BentoML framework has been used. In particular, service.py and bentofile.yaml are part of the depolyment with bentoml.

* *service.py* contains the logic for the prediction service.
* *bentofile.yaml* contains the dependencies.

To test the inference service in a local environment:

`bentoml serve service.py:svc --reload`

then the service can be tested through a swaggerUI interface in the browser

By executing:

`bentoml build`

a bento archive is built. The bento definition from the official docs is: 'Bento 🍱 is a file archive with all the source code, models, data files and dependency configurations required for running a user-defined bentoml.Service, packaged into a standardized format.'. In particular in the bento archive is included a Dockerfile

in order to containeraize the service from the bento archive:

`bentoml containerize midterm_classifier:xxxxxxxxxxxxxxx`

where midterm_classifier:xxxxxxxxxxxxxxx is the tag of the bento archive (as an example: bentoml containerize midterm_classifier:oah24sc6xgqjouon)

now to serve the prediction as a containerized service, execute:

`docker run -it --rm -p 3000:3000 midterm_classifier:oah24sc6xgqjouon`
(docker has to be installed to execute the last command.)

As a last note, I made this project in a Linux environment (Ubuntu 22.04) over a Windows OS (win11) by using WSL2 virtualization. Docker Desktop was installed in Windows and VSCode was used as IDE.

Hope you can enjoy!
