# Fake News Detection

## Team members
Wei Luo, Yanmin Ji, Yuanchu Dang

## Abstract
The spreading of rumours, or Fake News, keeps growing at an alarming rate on online social network platforms. With the recent booming development in Machine Learning fields, different models have been designed for human-level accuracy in specific tasks. However, most of them cannot be used for real-time predictions because of their large number of parameters.  This work presents a lightweight approach that has similar accuracy yet much faster predicting speed, making it an outstanding candidate for real-time analysis in big data era.

## Technical Details
Please see [here](report.pdf) for our report.

## Acknowledgement
* We forked and built on this [fake news challenge repo](https://github.com/uclmr/fakenewschallenge).
* [This NLP repo](https://github.com/dmlc/gluon-nlp/), particularly the Bert wrapper it provides. 
* Our demo is based on [this](https://github.com/naushadzaman/flask-socketio-with-twitter) twitter-flask multithreading repo. 

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
