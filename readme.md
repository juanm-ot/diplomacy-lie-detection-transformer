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

### Knowledge base
First, the associated research paper, [It Takes Two to Lie: One to Lie, and One to Listen](https://users.umiacs.umd.edu/~jbg/docs/2020_acl_diplomacy.pdf), is investigated and read to contextualize the project and identify starting points for knowledge evolution based on the authors' findings and iterations.

**Note:** The study document is also shared in its Spanish translated version: [Draft notes under "It Takes Two to Lie: One to Lie, and One to Listen"](https://jotalvaro-data-scientist.craft.me/eNyRE0M8jzGX9x).

The most relevant data from the project development are:

* The user study's training corpus maps language to truthfulness and deception annotations. The models incorporate conversational context and game power dynamics to approach human-level accuracy in detecting deception
* The results use a **weighted F1 metric for evaluating truth and lie predictions**, as accuracy can be inflated due to class imbalance. An approach is adopted during training where *incorrect predictions of lies are penalized more* than truthful statements, with the relative penalty between classes adjusted as a hyperparameter based on F1
* **They focus on lie detection**, where humans have an F1 score of 22.5 for lies
* One of the models used was **logistic regression**, as it offers interpretable coefficients that reveal linguistic phenomena related to lies. In the context of word usage, they propose integrating **Harbingers**, which are lists of words that can predict deception, though their coverage is limited as they focus on specific rhetorical areas. Nevertheless, a logistic regression model that includes all types of words as features can enhance the model's performance in terms of the F1 metric
* *Power dynamics influence language and the flow of conversation, potentially affecting the likelihood of lying*, as a more powerful player may feel more inclined to lie. Victory points reflect a player's performance, and the difference in power between players is measured by the difference in their victory points. This difference changes throughout the game and is captured by the power differential. Based on this theory, **the features "game score" and "game score delta" were created**
* **Neural Models vs. Logistic Regression:** Neural models, while often less interpretable than logistic regression models, generally offer greater accuracy. The authors utilized a standard Long Short-Term Memory (LSTM) network to explore whether sequences of words, which logistic regression might overlook, can reveal lies. By incorporating message context and power dynamics, they achieved improved accuracy over the base neural model.
* **Hierarchical LSTM and Contextual Focus:** A hierarchical LSTM can enhance the focus on specific phrases within lengthy conversational contexts. The authors argue that, similar to how it would be challenging for a human to detect a lie without prior context, methods that analyze individual messages alone are limited in the types of signals they can identify.
* The fine-tuning of BERT embeddings (Devlin et al., 2019) for this model did not lead to a notable improvement in F1, likely due to the relatively small size of our training data (Denis et al.,2020)

**Conclusion:** The hierarchical LSTM model approaches human performance in terms of F1 score by combining content with conversational context and power imbalance, achieving the best results for the goal of lie detection



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
