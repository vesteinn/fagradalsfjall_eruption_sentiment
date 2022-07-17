# Fagradalsfjall Eruption Sentiment Analysis of Tweets

## Introduction

This repository contains code and data to train a sentiment model on labeled data for Icelandic tweets.
In particular it is set up for analysis of tweets mentioning eruptions or earthquakes in relation to the Fagradallsfjall eruption in 2021.
Included are IDs for tweets which need to be fetched from Twitter, along with scripts to fine tune models and summarize results.
See below for results and further details.

## How it works

1. Fetch data using the Twitter API or hydrate using provided information
2. Filter data to confirm keywords are present as the API does not always return tweets with the keywords.
3. Train or use a sentiment analysis model capable of labeling Icelandic, a bi-lingual model trained on English sentiment data is used.
4. If the model only support positive and negative sentiment, modify the model to support neutral labels, this is the case here.
5. Fine tune the model on the labeled data with keywords masked out to prevent overfitting on the words themselves.
6. Label all data with the keywords masked out to see the full picture.
7. Summarize labeled data by week to draw histogram and for further analysis.

## The data

The data was collected from December 2019 to December 2021 by searching for text in Icelandic containing keywords related to eruptions and earthquakes, see `fagradals/keywords.py` for a file with the keywords used.

| Split    | Amount |
|----------|--------|
| Train    | 493    |
| Valid    | 121    |
| To label | 9727   |
| Total    | 10341  |

## Results

The performance of the trained model should be around F1 0.7 after four epochs

```bash
{'eval_loss': 1.0065765380859375, 'eval_accuracy': 0.7107438016528925, 'eval_f1': 0.7059344435633096, 'eval_precision': 0.7056612638007986, 'eval_recall': 0.7205136684303352, 'eval_by_category': {0: {'f1': 0.6461538461538462, 'precision': 0.6363636363636364, 'recall': 0.65625}, 1: {'f1': 0.75, 'precision': 0.6666666666666666, 'recall': 0.8571428571428571}, 2: {'f1': 0.7216494845360826, 'precision': 0.813953488372093, 'recall': 0.6481481481481481}}, 'eval_runtime': 0.4288, 'eval_samples_per_second': 282.201, 'eval_steps_per_second': 30.319, 'epoch': 4.0}
```

## Rehydrating and parsing data

Tweet ids with labels can be found in the `raw_to_hydrate` folder.
As tweets are not allowed to be distributed this is a necessary step to reproduce the work or otherwise make use of the data.
The files contain two columns, `id` and `label`.
Use a tool such as [Twarc](https://github.com/DocNow/twarc) to rehydrate them, you will need to have an API key from twitter to acces the data.
Write the text from the resulting tweets to `data/{train,valid,tolabel}.json` along with the label from the id file so that each record has the following format.

```bash
{"text": "Tweet goes here...", "label": {0,1,2}, "timestamp": "YYYY-MM-dd HH:MM:SS"}
```

I.e. you need to fetch the text information and timestamp data from the hydrated tweets. 

The label 0 is negative, 1 is positive and 2 is neutral.

As there are many ways to fetch tweets by ID, authentication is necessary and formats may change this step has not been scripted.

## How to reproduce

Make sure you have the necessary libraries set up, including CUDA if you want to train and evaluate on GPU.

```bash
conda create --name fagra python=3.8  # if you want to set up a new conda environment
conda activate fagra
#  Or follow the recommended setup for torch at https://pytorch.org/get-started/locally/
pip install torch transformers datasets sklearn
```

Then run

```bash
cd fagradals
python train.py > is_geo_twt_mask.out  # Trains the model
python label.py > labeled_tweets_sentiment_geo.tsv  # Labels all the data, including that used for training and evaluation to give the full picture
python make_histo.py labeled_tweets_sentiment_geo.tsv > labeled_tweets_sentiment_geo.histogram.tsv  # Prints statistics
```

The histogram data has the columns `week, neg, pos` and `neutral`.

## Citing

Bibtex to come.