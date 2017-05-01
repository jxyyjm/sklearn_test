#!/usr/bin/python
# -*- coding:utf-8 -*-

# @time      : 2017-05-01
# @author    : yujianmin
# @reference : 
# @what to-do: try a svm-model

from __future__ import print_function

import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
try: from sklearn.cross_validation import train_test_split
except: from sklearn.model_selection import train_test_split
## sklearn-version 0.20.0 will move the sklearn.cross_validation.train_test_split ##

class CTest_svm:
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

	def test_NuSVC(self):
		# Nu-SVC is similar to SVC(Support Vector Classification), 
		# but, uses a parameter to control the num of support vectors.
		clf    = svm.NuSVC(
				nu=0.4, \
				kernel='rbf', \
				degree=3.0, \
				gamma=0.001, \
				coef0=0.0, \
				probability=True, \
				shrinking=True, \
				tol=0.001, \
				cache_size=200.0, \
				verbose=False, \
				max_iter=600.0, \
				decision_function_shape='ovo', \
				random_state=100
				)
		# ====== parameters ====== #
		'''
			num : float, default=0.5, span (0, 1]
				  an upper bound on the fraction of training erros, 
				  an lower bound of the fraction of support vectors,
			kernel: string, default='rbf', 
					span ['linear', 'poly', 'rbf', 'sigmoid', precomputed'], or a callable
					kernel-function given for computing the kerneal-space,
					if a callable is given, it it used to pre-compute the kernel matrix
			degree: int, default=3
					degree of the polynomial kernel function ('kernel=poly'
					if kernel=others methods, ignored this setting.
			gamma : float, default='auto'
					kernel coefficient for 'rbf' 'poly' and 'sigmoid'
					if gamma=auto, kernel-coef=1/n_features
			coef0 : float, default=0.0
					independent term in kernel function
					it is only significant in 'poly' and 'sigmoid'
			probability: boolean, default=False
					True means enable probability-estimate.
						it must be enabled prior to calling 'fit', 
					and this will slow down 'fit' method.
			shrinking : boolean, default=True
					whether to use the shrinking heuristic
			tol   : float, default=1e-3
					tolerance of stopping criterion
			cache_size: float, optional
					specify the size of kernel cache (in MB)
			class_weight: {dict, 'balance'}, optional
					set the parameter C of class i to class_weight[i]*C for SVC.
					if not given, all classes_weight are 'one'.
					the 'balanced' model uses classes_weight as following:
					n_samples / (n_classes * np.bincount(y))
			verbose : boolean, default=False
					enable verbose output.
			max_iter: int, default=-1
					hard limit on iterations within solver,
					-1 means no limit.
			decision_function_shape : 'ovo' 'ovr' or None, default=None
					'ovo', one-vs-one decision function, 
					'ovo' get an function for each pair-class.
					 return a DF of shape (n_samples, n_classes*(n_classes-1)/2) as all other classifiers.
					'ovr', one-vs-rest decision function, 
					'ovr' get an function for each pair (class, the other class)
					 return a DF of shape (n_samples, n_classes) as all other classifiers.
			random_state : int, randomstate-instance, default=None
					when probability estimation, this setting is used 
					to shuffling the data (here will random generate seed') to do probability estimation.
		'''
		clf.fit(self.trainX, self.trainY)
		self.model = clf

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
		try:
			print ('support_', clf.support_.shape)
			print (clf.support_)
		except: print ('support_ getting wrong')
		try:
			print ('support_vectors_', clf.support_vectors_.shape)
			print (clf.support_vectors_)
		except: print ('support_vectors_ getting wrong')
		try:
			print ('n_support_', clf.n_support_.shape, 'sum is :', np.sum(clf.n_support_))
			print (clf.n_support_)
		except: print ('n_support_ getting wrong')
		try:
			print ('dual_coef_', clf.dual_coef_.shape)
			print (clf.dual_coef_)
		except: print ('dual_coef_ getting wrong')
		try:
			print ('coef_', clf.coef_.shape)
			print (clf.coef_)
		except: print ('coef_ getting wrong')
		try:
			print ('intercept_', clf.intercept_.shape)
			print (clf.intercept_)
		except: print ('intercept_ getting wrong')


if __name__=='__main__':
	TestModel = CTest_svm()
	TestModel.read_data()
	TestModel.test_NuSVC()
	TestModel.print_mid_para()
	TestModel.predict_test()
