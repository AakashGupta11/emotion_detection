## Abstract
The purpose of the project is to classify facial expressions or/and audio signals into one of the emotions (anger, disgust, fear, happy, neutral, sad, surprise). Both facial expressions and audio signals can be easily taken from user and plays vital role to determine emotion. The project has separate deep learning models to determine emotions based on facial expressions and audio signals. Both of the criteria then combined based on the probabilities of different emotions to predict final result.
Emotion detection has many practical applications. This project is made using python, keras and librosa.
 
## Introduction

As the exposure of machines with humans increase, the interaction also has to become smoother and more natural. In order to achieve this, machines have to be provided with a capability that let them understand the surrounding environment. Specially, the intentions of a human being. Nowadays, machines have several ways to capture their environment state trough cameras, microphone and sensors. Hence, using this information with suitable algorithms allow to generate machine perception. Emotion detection is necessary for machines to better serve their purpose. For example, the use of robots in areas such as elderly care or as porters in hospitals demand a deep understanding of the environment. Facial emotions
deliver information about the subject’s inner state. If a machine is able to obtain a sequence of facial images, then the use of deep learning techniques would help machines to be aware of their interlocutor’s mood.
It has the potential to become a key factor to build better interaction between humans and machines, while providing machines with some kind of self-awareness about its human peers, and how to improve its communication with natural intelligence.

Emotion can be recognized easily from facial expressions or speech signals. Emotional displays convey considerable information about the mental state of an individual. This has opened up a new research field called automatic emotion recognition, having basic goals to understand and retrieve desired emotions.
Several inherent advantages make facial expression and speech signals a good source for affective computing. For example, compared to many other biological signals (e.g., electrocardiogram), photo of faces and speech signals usually can be acquired more readily and economically.
Various techniques have been developed to find the emotions such as signal processing, machine learning, neural networks, computer vision.
 
Machine Learning (ML) is a subfield of Artificial Intelligence. It completely differs from other fields where any new feature has to be added by hand.

For instance, in software development, when a new requirement appears, a programmer has to create software to handle this new case. In ML, this is not exactly the case. The ML algorithms create models, based on input data. These models generate an output that is usually a set of predictions or decisions. Then, when a new requirement appears, the model might be able to handle it or to provide an answer without the need of adding new code.

ML is usually divided into 3 broad categories. Each category focuses on how the learning process is executed by a learning system. These categories are: supervised learning, unsupervised learning, and reinforcement learning.
Supervised learning has a set of tools focused on solving problems within its domain. One of those tools is called Artificial Neural Networks (ANN). Generally speaking, deep learning is a machine learning method that takes in an input X, and uses it to predict an output of Y. Given a large dataset of input and output pairs, a deep learning algorithm will try to minimize the difference between its prediction and expected output. By doing this, it tries to learn the association/pattern between given inputs and outputs — this in turn allows a deep learning model to generalize to inputs that it hasn’t seen before. Deep learning in neural networks is used for various tasks such image recognition, classification tasks, decision making, pattern recognition etc. Various other Deep Learning techniques such as multimodal deep learning used for feature extraction, image recognition made at ease.
 
## Problem Statement

Determine the emotions of the person given image of face or/and audio at that instant of time.
Both the image of face and audio can be easily retrieved and is practical in most of the scenarios. With the help of facial expression and speech signal the emotional state of person can be determined and can be used in various applications like:

•	In market research to observe user’s reaction while interacting with a brand or a product or in the case of e-learning platforms.
•	In the case of call center, determine anger and stress levels in the voice and prioritize angry calls.
•	In-car board system based on information of the mental state of the driver can be provided to the system to initiate his/her safety preventing accidents to happen.
•	Recommendations based on emotions like music recommendation system.

Classifying an image based on its depiction can be a complicated task for machines. Several human emotions can be distinguished only by subtle differences in facial patterns, with emotions like anger and disgust often expressed in very similar ways. Constraints like low latency requirement should also be considered.
 
## Workflow
1.	Collecting dataset
Collecting data of facial expressions and speech signals from different resources.

2.	Exploratory Data Analysis
Perform EDA on image and audio dataset to gain some knowledge about the data i.e., how data is represented, counting number of images and audios for different labels, determine whether under-sampling or oversampling is required or not etc.

3.	Preprocessing and Feature extraction
Variations that are irrelevant to facial expressions, such as different backgrounds, illuminations, are fairly common in unconstrained scenarios. Therefore, before training the deep neural network to learn meaningful features, pre-processing is required. Converting images to gray scale and extracting the face from the image using haar cascade algorithm of OpenCV and transformations like resizing and scaling.
Extract features from audio like MFCC, pitch, spectral centroid etc.

4.	Splitting
Splitting data into train, test and cross validation data.

5.	ML algorithms
Applying different deep learning algorithms, with different features, like ANN or CNN to train the model. Experimenting with different features and different algorithms on training and cross validation dataset.

6.	Determining Performance
Determining performance of different models using various performance metrics to determine best approach.

7.	Combining Probabilities
Assigning weights based on importance of criteria and then give final result by combining the probabilities.
 
## Algorithms
### Haar Cascade
Haar Cascade is a machine learning object detection algorithm proposed by Paul Viola and Michael Jones. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. The idea of Haar cascade is extracting features from images using a kind of ‘filter’, similar to the concept of the convolutional kernel. These filters are called Haar features and look like that:

### Artificial Neural Network
Artificial Neural Networks (ANN) are multi-layer fully-connected neural nets. They consist of an input layer, multiple hidden layers, and an output layer. Every node in one layer is connected to every other node in the next layer. A given node takes the weighted sum of its inputs, and passes to an activation function. This is the output of the node, which then becomes input of another node in the next layer. Backpropagation algorithm is used to efficiently train artificial neural networks following a gradient-based optimization algorithm that exploits the chain rule.
 
### Convolution Neural Network
A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.

 
## Datasets Used

### Facial expressions

•	Kaggle Dataset: This is the final project for DATA 622, Fall 2016 at CUNY MS Data Analytics. The data has come from a variety of sources like Labeled faces in the wild, IMFDB etc.
•	FaceDB: The database IMPA-FACE3D was created in 2008 to assist in the research of facial animation. This dataset includes acquisitions of 38 individuals each having with a neutral face sample, samples corresponding to six universal facial expressions.
•	Jaffe: The Japanese Female Facial Expression (JAFFE) database is a laboratory-controlled image database that contains 213 samples of posed expressions from 10 Japanese females.
•	CK+: The Extended CohnKanade (CK+) database is the most extensively used laboratory-controlled database for evaluating FER systems.There are 593 sequences across 123 subjects which are FACS coded at the peak frame. All sequences are from the neutral face to the peak expression.
•	FERG-DB: Facial Expression Research Group Database (FERG-DB) is a database of stylized characters with annotated facial expressions. The database contains 55767 annotated face images of six stylized characters.
•	FER2013: The FER2013 database was introduced during the ICML 2013 Challenges in Representation Learning. FER2013 is a large-scale and unconstrained database collected automatically by the Google image search API.

### Audio signals

•	SAVEE: Surrey Audio-Visual Expressed Emotion (SAVEE) database has been recorded as a pre-requisite for the development of an automatic emotion recognition system.
•	RAVDESS: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent.
•	TESS: A set of 200 target words were spoken in the carrier phrase "Say the word	' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions.