# Diplomacy Lie Detector: A Transformer-Based Approach

## Index
1. [Introduction](#introduction)
2. [Solution strategy](#solution-strategy)
3. [Technical solution](#technical-solution)
4. [Project structure](#project-structure)
5. [Configuration](#configuration)


## Introduction
### Context
The game of Diplomacy is a strategic board game that simulates political and military conflict in Europe during World War I. Players take on the roles of leaders of various nations, making diplomatic and military decisions to achieve their goals while contending with betrayal and negotiation from opponents. Due to its complexity and the need for subtle interactions, Diplomacy has become an intriguing field for analyzing communication and detecting deception.

In the context of natural language processing (NLP), the goal is to develop a deep learning model to predict whether a message in the game of Diplomacy is a lie or not. This project will utilize transformer-based models and transfer learning techniques, which have proven highly effective for NLP tasks.

* **Transformers** are a class of deep learning models that have revolutionized NLP by efficiently handling long and complex contexts. Their architecture allows for capturing relationships and patterns in textual data effectively.

* **Transfer Learning** is a technique where a model pre-trained on a related task is fine-tuned for a specific task. This method is particularly useful when limited data is available for the specific task, as it leverages the knowledge acquired from previous tasks.

### Project Goal 
The goal of this project is to understand the Diplomacy dataset, conduct exploratory data analysis (EDA), and apply advanced NLP techniques using transformer-based models. The initial focus will be on understanding the characteristics of the data and the technological capabilities available in terms of models and tools. Subsequent efforts will involve iterative training and fine-tuning of the model within a notebook environment, aiming to develop a robust product for lie detection.

The performance of the trained model will be compared with the benchmark provided in the ACL’20 paper by Peskov et al. to evaluate its effectiveness and adjust approaches as necessary


## Solution Strategy

## Technical Solution

## Project structure

```linux

.
├── data                               # data storage in stages
│   │── raw                            # stores JSON 
│   └── intermediate                   # stores df with data extracted from JSON
├── src                                # contains the core functionality of the project
│   │── utils.py                       #       
│   └── functions.py                   # 
├── exploring                          # contains data and modeling exploration
│   └── notebook.ipynb 
├── resources                          # folder: contains no binary files for docs
├── .env                               # contains the environment vars
├──                   
│
├── README.md                          # project documentation
└── requirements.txt                   

```

## Configuration
