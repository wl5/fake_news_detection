"""
Big Data Fake News Project
Wei Luo, Yuanchu Dang, Yanmin Ji
Replace the keys with your actual keys.
"""

# first make sure we have the correct asynchronous mode
async_mode = None

if async_mode is None:
    try:
        import eventlet
        async_mode = 'eventlet'
    except ImportError:
        pass

    if async_mode is None:
        try:
            from gevent import monkey
            async_mode = 'gevent'
        except ImportError:
            pass

    if async_mode is None:
        async_mode = 'threading'

    print('async_mode is ' + async_mode)

# monkey patching is necessary because this application uses a background
# thread
if async_mode == 'eventlet':
    import eventlet
    eventlet.monkey_patch()
elif async_mode == 'gevent':
    from gevent import monkey
    monkey.patch_all()

# begin main program
import json 
import time
from threading import Thread
from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

from pyspark import SparkContext, SparkConf

    
from tweepy.streaming import StreamListener
from tweepy import Stream
import tweepy
from biLSTM import Model
#from BOW_MLP import Model

# setup flask and socket
app = Flask(__name__)
socketio = SocketIO(app, async_mode=async_mode)
thread = None

# keys for tweepy
cred = {
            "consumer_key": "", 
            "consumer_secret": "", 
            "access_key": "", 
            "access_secret": ""
        }
auth = tweepy.OAuthHandler(cred['consumer_key'], cred['consumer_secret'])
auth.set_access_token(cred['access_key'], cred['access_secret'])

# set up spark
conf = SparkConf().setAppName('demo_elephas').setMaster('local[2]')
sc = SparkContext(conf=conf)

# load trained model
model = Model()
#model.load_model("BOW_MLP003.h5")
model.load_model("biLSTM030.h5")
model.model = SparkModel(model.model)
# make any preliminary prediction to put model prediction on correct thread
res = model.predict(["a"], "b")
classes = ["agree", "disagree", "discuss", "unrelated"]

# global pred_body.  this will be set when user inputs one
global pred_body
pred_body = ""

def make_prediction(pred_head):
    """
    Make predictions using our trained model.
    Returns an element of classes list.
    """
    print("predicting---------------------------------")
    print("head is ", pred_head)
    print("body is ", pred_body)

    res = model.predict([pred_head], pred_body)
    print(classes[res[0]])
    return classes[res[0]]

class StdOutListener(StreamListener):
    """
    Custom listener for stream
    """
    def __init__(self):
        pass 
        
    def on_data(self, data):
        try: 
            tweet = json.loads(data)
            # make prediction and emit result to front-end
            pred = make_prediction(tweet['text'])
            socketio.emit('stream_channel',
                  {'pred': pred},
                  namespace='/demo_streaming')
            print(text)
        except: 
            pass 

    def on_error(self, status):
        print('Error status code', status)
        exit()


def background_thread():
    """
    Background thread that runs the streaming
    """
    stream = Stream(auth, l)
    _keywords = ["Michael Cohen"]
    stream.filter(languages=["en"], track=_keywords) 


@app.route('/submit', methods = ['POST'])
def submit():
    global pred_body
    pred_body = request.form['news_body']
    print(pred_body)
    global thread
    if thread is None:
        thread = Thread(target=background_thread)
        thread.daemon = True
        thread.start()
    return render_template('index.html')
    
@app.route('/')
def index():
    return render_template('index.html')


l = StdOutListener()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port='8888')
