# -*- coding: utf-8 -*-
'''
Created on Wed May 29 14:21:54 2019 by Attila Lengyel - attila@lengyel.nl

Calculates IoU scores of dataset per individual batch.
Arguments:
    - labelNames: list of class names
    - validClasses: list of valid class ids
    - voidClass: class id of void class (class ignored for calculating metrics)
'''

import sys
import numpy as np

class iouCalc():
    
    def __init__(self, classLabels, validClasses, voidClass = None):
        assert len(classLabels) == len(validClasses), 'Number of class ids and names must be equal'
        self.classLabels    = classLabels
        self.validClasses   = validClasses
        self.voidClass      = voidClass
        self.evalClasses    = [l for l in validClasses if l != voidClass]
        
        self.perImageStats  = []
        self.nbPixels       = 0
        self.confMatrix     = np.zeros(shape=(len(self.validClasses),len(self.validClasses)),dtype=np.ulonglong)
        
        # Init IoU log files
        self.headerStr = 'epoch, '
        for label in self.classLabels:
            if label.lower() != 'void':
                self.headerStr += label + ', '
            
    def clear(self):
        self.perImageStats  = []
        self.nbPixels       = 0
        self.confMatrix     = np.zeros(shape=(len(self.validClasses),len(self.validClasses)),dtype=np.ulonglong)
        
    def getIouScoreForLabel(self, label):
        # Calculate and return IOU score for a particular label (train_id)
        if label == self.voidClass:
            return float('nan')
    
        # the number of true positive pixels for this label
        # the entry on the diagonal of the confusion matrix
        tp = np.longlong(self.confMatrix[label,label])
    
        # the number of false negative pixels for this label
        # the row sum of the matching row in the confusion matrix
        # minus the diagonal entry
        fn = np.longlong(self.confMatrix[label,:].sum()) - tp
    
        # the number of false positive pixels for this labels
        # Only pixels that are not on a pixel with ground truth label that is ignored
        # The column sum of the corresponding column in the confusion matrix
        # without the ignored rows and without the actual label of interest
        notIgnored = [l for l in self.validClasses if not l == self.voidClass and not l==label]
        fp = np.longlong(self.confMatrix[notIgnored,label].sum())
    
        # the denominator of the IOU score
        denom = (tp + fp + fn)
        if denom == 0:
            return float('nan')
    
        # return IOU
        return float(tp) / denom
    
    def evaluateBatch(self, predictionBatch, groundTruthBatch):
        # Calculate IoU scores for single batch
        assert predictionBatch.size(0) == groundTruthBatch.size(0), 'Number of predictions and labels in batch disagree.'
        
        # Load batch to CPU and convert to numpy arrays
        predictionBatch = predictionBatch.cpu().numpy()
        groundTruthBatch = groundTruthBatch.cpu().numpy()

        for i in range(predictionBatch.shape[0]):
            predictionImg = predictionBatch[i,:,:]
            groundTruthImg = groundTruthBatch[i,:,:]
            
            # Check for equal image sizes
            assert predictionImg.shape == groundTruthImg.shape, 'Image shapes do not match.'
            assert len(predictionImg.shape) == 2, 'Predicted image has multiple channels.'
        
            imgWidth  = predictionImg.shape[0]
            imgHeight = predictionImg.shape[1]
            nbPixels  = imgWidth*imgHeight
            
            # Evaluate images
            encoding_value = max(groundTruthImg.max(), predictionImg.max()).astype(np.int32) + 1
            encoded = (groundTruthImg.astype(np.int32) * encoding_value) + predictionImg
        
            values, cnt = np.unique(encoded, return_counts=True)
        
            for value, c in zip(values, cnt):
                pred_id = value % encoding_value
                gt_id = int((value - pred_id)/encoding_value)
                if not gt_id in self.validClasses:
                    printError('Unknown label with id {:}'.format(gt_id))
                self.confMatrix[gt_id][pred_id] += c
        
            # Calculate pixel accuracy
            notIgnoredPixels = np.in1d(groundTruthImg, self.evalClasses, invert=True).reshape(groundTruthImg.shape)
            erroneousPixels = np.logical_and(notIgnoredPixels, (predictionImg != groundTruthImg))
            nbNotIgnoredPixels = np.count_nonzero(notIgnoredPixels)
            nbErroneousPixels = np.count_nonzero(erroneousPixels)
            self.perImageStats.append([nbNotIgnoredPixels, nbErroneousPixels])
            
            self.nbPixels += nbPixels
            
        return
            
    def outputScores(self, epoch=None):
        # Output scores over dataset
        assert self.confMatrix.sum() == self.nbPixels, 'Number of analyzed pixels and entries in confusion matrix disagree: confMatrix {}, pixels {}'.format(self.confMatrix.sum(),self.nbPixels)
        # TODO: Add accuracy
        # Calculate IOU scores on class level from matrix
        classScoreList = []
            
        # Print class IOU scores
        classLog = ''
        outStr = '---------------------\n'
        outStr += 'Epoch {}\n'.format(epoch)
        outStr += 'classes           IoU\n'
        outStr += '---------------------\n'
        for c in self.evalClasses:
            iouScore = self.getIouScoreForLabel(c)
            classScoreList.append(iouScore)
            classLog += '{:.3f}, '.format(iouScore) # append scores to single line in log file
            outStr += '{:<14}: {:>5.3f}\n'.format(self.classLabels[c], iouScore)
        miou = getScoreAverage(classScoreList)
        outStr += '---------------------\n'
        outStr += 'Score Average : {avg:5.3f}\n'.format(avg=miou)
        outStr += '---------------------\n'
        
        return miou, outStr, self.confMatrix
        
# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)

def getScoreAverage(scoreList):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not np.isnan(score):
            validScores += 1
            scoreSum += score
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores
