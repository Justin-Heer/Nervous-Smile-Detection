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

Tensorflow version $>=$ 2.2 is required, the rest of the libraries used are listed
in the requirements.txt file included in the repository. The reccommended method
to setup your environment is to use Anaconda which will read the requirements.txt
file and construct the environment accordingly


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
    |--VGG-face+BiLSTM.py # The main NN trained for our project
    |
    |--



<!-- Self Evaluation -->
## Self Evaluation


