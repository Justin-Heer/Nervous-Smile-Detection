<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
* [Structure](#Structure)
* [Self Evaluation](#Self-Evaluation)



<!-- ABOUT THE PROJECT -->
## About The Project

There are many great smile detectors out there, however, we could not find one capable of 
discerning between nervous and happy smiles.

We have curated a dataset from scratch that contains happy and nervous smiles. 
With this we were able to train both an SVM and a CNN+BiLSTM NN capable of distinuishing 
between nervous and happy smiles. Our medium blog post walks you through most of the 
iterative process that we went through when working on this project

We hope you will enjoy looking at our work!


### Built With
* Python
* C++ (compiler)
* FFMPEG
* YouTube-dl
* OpenFace


<!-- GETTING STARTED -->
## Getting Started

The CNN+BiLSTM.py script has created to run both with and without training the CNN.
The output from the CNN has been previously trained by the researchers so that 
you would not have to do this step again, trainign the CNN takes considerable time.
However, if you wish to retrain the CNN anyway for your own personal reasons, feel 
free to delete the datasets.npz file that is included in the repository. 
This will trigger the script to train the CNN automatically. Do note however, if
you wish to train the CNN you will require a C++ compiler of your choice, we 
reccommend gcc with g++ for windows users. MacOS users...you're on your own here

We have already extracted the OpenFace outputs and included them in the repository
thus OpenFace is not required to run any scripts

Tensorflow version >= 2.2 is required, the rest of the libraries used are listed
in the requirements.txt file included in the repository. The reccommended method
to setup your environment is to use Anaconda which will read the requirements.txt
file and construct the environment accordingly

NVIDIA GPU WARNING: If you do not have an NVIDIA GPU you will need to comment out the 
tf.config.experimental.set_memory_growth(gpu[0], True) lines and any others
that cause an error because of it. We have tested all scripts on both AMD and
NVIDIA systems and as long the lines giving the error are commented out the 
script runs perfectly, although it takes much longer with GPU access.

<!-- Structure -->
## Structure

    CMPT419-term-project
    |
    |--dataset
    |       |
    |       |--happy_frames
    |       |       |
    |       |       |--_0/ #folder with images for happy training ex 0
    |       |       |  .
    |       |       |  .
    |       |       |  .
    |       |       |--_227/ #folder with images for happy training ex 227
    |       |                       
    |       |--happy_frames_openface # Contains OpenFace output for all happy training ex
    |       |
    |       |--nervous_frames 
    |       |       |
    |       |       |--_0/ #folder with images for nerv training ex 0
    |       |       |  .
    |       |       |  .
    |       |       |  .
    |       |       |--_115/ #folder with images for nerv training ex 115
    |       |                       
    |       |--nervous_frames_openface # Contains OpenFace output for all nerv training ex    
    |
    |--gitignore #tells git what to ignore for source control
    |
    |--README.md #This file that you are reading
    |
    |--SVM-baseline.ipynb # This is our baseline classifier
    |
    |--VGG-face+BiLSTM.py # The main NN trained for our project
    |
    |--action-unit-occurence-by-smile.png # An image that is plotted
    |
    |--dataset-human-accuracy-score-generator.py #Dataset annotation script
    |
    |--dataset-human-accuracy-score-tester.py # tests output of generator
    |
    |--exploratory-analysis.ipynb #Showcases a simple exploration into the dataset
    |
    |--frame_extraction_util.ipynb #Used by researchers to extract frames


<!-- Self Evaluation -->
## Self Evaluation

Our algorithm is able to differeniate between nervous and happy smiles, but
not with great accuracy as we had hoped. We were able to use OpenFace, an SVM 
and a CNN+BiLSTM NN. We were not able to test a TCN due to time constraints. 
Our NN model was able to do better than our baseline SVM model however it did 
not do as well as we had hoped. We considered the prospect of quickly throwing
together a TCN, however we thought it would be better to perform 
hyperparameter tuning on the model we had already built. Our project has 
undergone considerable change over the course of the past couple months, what
you see now is just the final product not all the steps it took to get here.        




