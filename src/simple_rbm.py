#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-06-18
# @author    : yujianmin
# @reference : 
# @what to-do: try a rbm-model

from __future__ import print_function

import numpy as np
from sklearn import metrics
from sklearn import datasets
from sklearn import neural_network
try: from sklearn.cross_validation import train_test_split
except: from sklearn.model_selection import train_test_split
## sklearn-version 0.20.0 will move the sklearn.cross_validation.train_test_split ##

class CTest_RBM:
	def __init__(self):
		self.trainX = ''
		self.trainY = ''
		self.testX  = ''
		self.testY  = ''
		self.model  = ''
	def __del__(self):
		self.trainX = ''
		self.trainY = ''
		self.testX  = ''
		self.testY  = ''
		self.model  = ''

	def read_data(self):
		digits = datasets.load_digits()
		trainX, testX, trainY, testY = train_test_split(
										digits.data,   digits.target, \
										test_size=0.2, random_state=100)
		print ('trainX :', trainX.shape, type(trainX))
		print ('trainY :', trainY.shape, type(trainY))
		self.trainX = trainX
		self.trainY = trainY
		self.testX  = testX
		self.testY  = testY

	def test_RBM(self):
		'''
			class BernoulliRBM
			description : Bernoulli Restricted Bolltzmann Machine.
			1) Parameters
			n_components : int, Num-of binary hidden unitis
			learning_rate : float, range 10**[0, -3]
			batch_size : num-of examples each minibatch
			n_iter : int, num-of iterations for training
			verbose: int, 
			random_state : integer or np.RandomState
			2) Attributes
			intercept_hidden_ : array-like, bias of the hidden units. shape=(n_components, )
			intercept_visible_ : array-like, bias of the visible units, shape=(n_features, )
			components_ : array-like, bias of weight matrix. shape=(n_components, n_features)
			3) Methods
			fit(X, Y=None)
				Fit the model to the data X.
			gibbs(v)
				perform one Gibbs sample step.
				v : array-like, shape=(n_samples, n_features)
					values of visible layer to start from
				return: v_new, array-like, shape is same.
					value of visible layers after one Gibbs step.
			partial_fit(X, y=None)
				Fit the model to the data X, which should contain a segment of the data
			score_samples(X)
				compute the pseudo-likelihood of X.
				X : shape=(n_samples, n_features)
					values of the visible-layer.
					must be all-boolean. not-check.
				return, pseudo-likelihood.
					shape=(n_samples, )
			transform(X)
				compute the hidden-layer activation prob.
				X : shape=(n_samples, n_features)
				return, shape=(n_samples, n_components)
			get_params(deep=True)
			set_params()
			========================================================================================
			class :: MLPClassifier
		'''
		n_hidden = 100
		clf      = BernoulliRBM(
					n_components = n_hidden,
					learning_rate= 0.1,
					batch_size   = 100,
					n_iter       = 300,
					random_state = None
					)
		Train_XY = np.hstack((self.trainX, self.trainY))
		clf.fit(Train_X)
		self.model   = clf

	def predict_test(self):
		pred_test  = self.model.predict(self.testX)
		pred_train = self.model.predict(self.trainX)
		print ('train accuracy :', metrics.accuracy_score(self.trainY, pred_train), \
			   'test  accuracy :', metrics.accuracy_score(self.testY,  pred_test)) 
		
		print ('train confusion matrix :')
		print (metrics.confusion_matrix(self.trainY, pred_train))

		print ('test confusion matrix :')
		print (metrics.confusion_matrix(self.testY, pred_test))

	def print_mid_para(self):
		clf = self.model


if __name__=='__main__':
	TestModel = CTest_RBM()
	TestModel.read_data()
	TestModel.test_RBM()
	TestModel.print_mid_para()
	TestModel.predict_test()
	print ('last think of a question, how to CV-test, and find the best para')
