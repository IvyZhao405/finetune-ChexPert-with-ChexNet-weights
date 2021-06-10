# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:50:51 2020

@author: User
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from src import config as cfg


OUT_DIR = "../output/"

label_names=['Atelectasis',
'Cardiomegaly',
'Cosolidation',
'Edema',
'PleuralEffusion']

def plot_training_loss_curves(train_loss, val_loss):
    """
    train_loss: np.array or list, training loss values

    val_loss: np.array or list, validation loss values

    """
    output = os.path.join(OUT_DIR, 'training_loss_curves.png')
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss, 'bo', label='Training loss')
    plt.plot(range(len(val_loss)), val_loss, 'b', label='Validation loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(output)


def plot_training_accuracy_curves(train_acc, val_acc):
    """
    train_acc: np.array or list,, training accuracy values

    val_acc: np.array or list, validation accuracy values

    """
    output = os.path.join(OUT_DIR, 'training_accuracy_curves.png')
    plt.figure()
    plt.plot(range(len(train_acc)), train_acc, 'bo', label='Training Accuracy')
    plt.plot(range(len(val_acc)), val_acc, 'b',    label='Validation Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(output)



def plot_confusion_matrix_custom(y_true,y_pred, label,approach):



	normalize = True
	title = False
	cmap = plt.cm.Blues
	class_names =['-ve','+ve']
	model_dir = cfg.output_path[approach]['directory']

	"""
	    This function prints and plots the confusion matrix.
	    Normalization can be turned off by setting `normalize=False`.
	    """

	if not title:
		if normalize:
			title = approach + '--Normalized confusion matrix --'+ label
		else:
			title = approach + '--Confusion matrix, without normalization--'+ label

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		#print("Normalized confusion matrix")
	else:
		pass
		#print('Confusion matrix, without normalization')

	#print(cm)
	print('LABEL: ',label)
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=class_names, yticklabels=class_names,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	figname = 'output/'+model_dir+'/'+approach+'- Confusion_Matrix '+ '- '+label+ '.png'
	plt.savefig(figname)
	plt.show()



def plot_auroc_curve(result, output_name,approach):
    """
    Result: pandas dataframe
    output_name: String, name of the out file

    Output: AUROC curves

    Calculates the AUROC scores as done in the original paper for each label
    and creates a plot for each label visualizing the score
    """
    model_dir = cfg.output_path[approach]['directory']
    #If we change the output of result, remember to change this here.
    y_pred = np.array(result.iloc[:,0:5])

    y_test = np.array(result.iloc[:,5:10])

    #If we split result inyo y_test and y_pred before this function, then
    #uncomment out the part below and change the header of this function

#
#    if type(y_test) is not np.ndarray:
#        y_test = np.array(y_test)
#
#    if type(y_pred) is not np.ndarray:
#        y_pred = np.array(y_pred)

   # output = os.path.join(OUT_DIR, output_name)
    output ='output/'+model_dir+'/'+output_name

    plt.figure(figsize=(20,10))
    for index, label_name in enumerate(label_names):
        y_true = y_test[:,index]
        y_score = y_pred[:,index]

        fpr, tpr, thresh = roc_curve(y_true, y_score)
        auroc = auc(fpr, tpr)

        plt.subplot(3, 5, index + 1)
        plt.plot(fpr, tpr, color='b', lw=2, label='AUROC = %0.2f)' % auroc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(label_name)
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output)

def plot_precision_recall_curve(result, output_name,approach):
    """
    Result: pandas dataframe
    output_name: String, the name of the out file

    Output: PRAUC curves

    Calculates Precision-Recall AUC Scores like done in the original
    chexpert paper, and makes a plot for each label visualizing the scores.
    """
    model_dir = cfg.output_path[approach]['directory']
    #If we change the output of result, remember to change this here.
    y_pred = np.array(result.iloc[:,0:5])

    y_test = np.array(result.iloc[:,5:10])

    #If we split result inyo y_test and y_pred before this function, then
    #uncomment out the part below and change the header of this function

#
#    if type(y_test) is not np.ndarray:
#        y_test = np.array(y_test)
#
#    if type(y_pred) is not np.ndarray:
#        y_pred = np.array(y_pred)

    #output = os.path.join(OUT_DIR, output_name)
    output ='output/'+model_dir+'/'+output_name

    plt.figure(figsize=(20,10))
    for index, label_name in enumerate(label_names):
        y_true = y_test[:,index]
        y_score = y_pred[:,index]

        precision, recall, thresh = precision_recall_curve(y_true, y_score)
        prauc = auc(recall, precision)

        plt.subplot(3, 5, index + 1)
        plt.plot(recall, precision, color='b', lw=2, label='PRAUC = %0.2f)' % prauc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.0])
        plt.xlabel('Sensitivity')
        plt.ylabel('Precision')
        plt.title(label_name)
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(output)
