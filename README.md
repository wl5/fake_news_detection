# Fake News Detection

## Team members

* [Yuanchu Dang](https://www.linkedin.com/in/yuanchu-dang-6364a562/) - Bert model
* Yanmin Ji - LSTM model
* Wei Luo - Demo using multithreaded Tweepy streaming and Flask 

## Previous work & Acknowledgement
* We forked and built on this [fake news challenge repo](https://github.com/uclmr/fakenewschallenge).
* [This NLP repo](https://github.com/dmlc/gluon-nlp/), particularly the Bert wrapper it provides. 
* Our demo is based on [this](https://github.com/naushadzaman/flask-socketio-with-twitter) twitter-flask multithreading repo. 

## Problem Formulation
Given a headline and a body concerning a subject, determine if their views agree or disagree with each other, or, are simply unrelated. 

## Approaches


## Running instruction
To run the demo, you need to first fill in the dict that contains API keys with your actual keys.  Also, in the main function, you can specify which port you want Flask to use.  Then to run the application, you can simply run 
```
python3 demo.py
```
Then you can go to 
```
http://[Your_External_IP]:[Port]
```
to access the demo webpage.
