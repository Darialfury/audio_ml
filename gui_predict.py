import numpy as np
import pickle
import pyaudio
import sys
import wave

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior

def load_object(file_path):
    """Load file in a object"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


class customSoftmaxClassifier(object):
    def __init__(self, X, Y, X_val=None, Y_val=None, lr=1e-1, reg=0,
                 num_iter=2500, batch_size=100, scale_init=1e-3):
        self.x_train = X
        self.y_train = Y
        self.x_val = X_val
        self.y_val = Y_val
        self.lr = lr
        self.reg = reg
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.loss_history = list()
        self.train_acc = list()
        self.loss_val_history = list()
        self.val_acc = list()
        self.W = np.random.randn(X.shape[1], np.max(Y) + 1) * scale_init
    def softmax_loss_vectorization(self, W, X, y, reg):
        """
        Softmax loss function

        D: number of features at the input vector
        C: number of classes in the dataset
        N: number of samples to operate on minibatches

        Inputs
         W: A numpy array of shape (D, C) containing weights.
         X: A numpy array of shape (N, D) containing a minibatch of data.
         y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
         reg: (float) regularization strength

        Output:
         loss: the total loss as single float
         dW: gradients with respect to weights W; an array of same shape as W
        """
        loss = 0
        dW = np.zeros(W.shape)

        ##########################################################################
        # TODO: compute loss and gradients for this function, take into account  #
        # regularization in the loss and the gradients                           #
        # START                                                                  #
        ##########################################################################

        num_train = X.shape[0]
        num_classes = np.max(y) + 1
        y_one_hot = np.zeros([num_train, num_classes])
        y_one_hot[np.arange(num_train), y] = 1

        dot_xw = np.dot(X, W)
        # numerical stability
        max_score = np.max(dot_xw, axis=1).reshape(-1, 1)
        stabilized_xw = dot_xw - max_score
        exp_stabilized_xw = np.exp(stabilized_xw)
        matrix_q = exp_stabilized_xw/np.sum(exp_stabilized_xw, axis=1).reshape(-1, 1)

        # compute loss
        vec_exp = exp_stabilized_xw[np.arange(num_train), y].reshape(-1, 1)
        sum_exp = np.sum(exp_stabilized_xw, axis=1).reshape(-1, 1)

        loss = (-1/num_train)*np.sum(np.log(vec_exp/sum_exp)) + reg*np.sum(W*W)

        # compute
        xmat = np.zeros_like(dot_xw)
        xmat[np.arange(num_train), y] = -1
        grad = (1/num_train)*(np.dot(X.T, xmat) + np.dot(X.T, matrix_q)) + 2*reg*W

        ##########################################################################
        # END                                                                    #
        ##########################################################################
        return loss, grad
    def train(self):
        num_train = self.x_train.shape[0]
        for k in range(self.num_iter):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Implement SGD (stochastic gradient descent)                           #
            # Hint: look for the function in numpy np.random.choice                 #
            #########################################################################
            # randomly sample some samples
            #sample_indices = np.random.choice(np.arange(num_train), self.batch_size)
            # x_batch = self.x_train[sample_indices]
            # y_batch = self.y_train[sample_indices]

            # evaluate loss and gradient
            loss, grad = self.softmax_loss_vectorization(self.W, self.x_train, self.y_train, self.reg)
            self.loss_history.append(loss)
            pred = self.predict(self.x_train)
            self.train_acc.append(np.mean(pred == self.y_train))
            if self.x_val is not None:
                l, _  = self.softmax_loss_vectorization(self.W, self.x_val, self.y_val, self.reg)
                self.loss_val_history.append(l)
                pred = self.predict(self.x_val)
                self.val_acc.append(np.mean(pred == self.y_val))


            # perform parameter update
            self.W = self.W - self.lr * grad
            #########################################################################
            #  END                                                                  #
            #########################################################################
    def predict(self, X):
        # TODO: Implement prediction function for svm
        #   X: input array with shape (N_test x D)
        return np.argmax(np.dot(X, self.W), axis=1)
    def compute_prob(self, X):
        # return probabilities from a sample
        #   X: input array with shape (N_test x D)
        dot_xw = np.dot(X, self.W)
        max_score = np.max(dot_xw, axis=1).reshape(-1, 1)
        stabilized_xw = dot_xw - max_score
        exp_stabilized_xw = np.exp(stabilized_xw)
        matrix_q = exp_stabilized_xw/np.sum(exp_stabilized_xw, axis=1).reshape(-1, 1)
        return matrix_q
    def get_loss(self):
        return self.loss_history
    def get_loss_val(self):
        return self.loss_val_history
    def get_train_acc(self):
        return self.train_acc
    def get_val_acc(self):
        return self.val_acc
    def getW(self):
        return self.W


class Recorder(object):
    '''A recorder class for recording audio to a WAV file.
    Records in mono by default.
    '''

    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                            self.frames_per_buffer)

class RecordingFile(object):
    def __init__(self, fname, mode, channels, 
                rate, frames_per_buffer):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self.get_callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile


class TestApp(App):
    def build(self):
        self.dict_model = load_object("classifier.pkl")
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=44100,
                                        input=True,
                                        frames_per_buffer=1024)

        parent = Widget()
        rec_button = Button(text="Make identification")
        rec_button.bind(on_release=self.recognize_me)
        parent.add_widget(rec_button)
        return parent


    def recognize_me(self, obj):
        # Record audio file
        rec = Recorder(channels=2)
        duration = 2
        filename = "default.wav"
        with rec.open(filename, 'wb') as recfile:
            recfile.record(duration=2.0)
        print("recorded file")

        # read audio file
        wf = wave.open(filename, 'rb')
        data = wf.readframes(1024)

        # process audio file
        new_x = np.frombuffer(data, np.int16).astype(np.float64) - self.dict_model["mean"]
        new_x = np.hstack([new_x.reshape(1, -1), np.ones([1, 1])])
        print(new_x.shape)
        prob = self.dict_model["model"].compute_prob(new_x.reshape(1, -1))
        class_pred = np.argmax(prob, axis=1)
        if class_pred[0] == 0:
            print("not me")
        else:
            print("yep its me")
        print(prob[0][1])

TestApp().run()
