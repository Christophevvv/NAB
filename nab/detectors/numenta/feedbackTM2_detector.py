# ----------------------------------------------------------------------
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import math
import numpy as np
from nupic.algorithms import anomaly_likelihood, anomaly
from nab.detectors.base import AnomalyDetector
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.encoders.date import DateEncoder
from corticalcolumn_new import CorticalColumn
from nupic.bindings.algorithms import TemporalMemory
import cv2

SPATIAL_TOLERANCE = 0.05

class FeedbackTM2Detector(AnomalyDetector):
  """
  This detector uses a research version of HTM where apical feedback is added.
  """

  def __init__(self, *args, **kwargs):

    super(FeedbackTM2Detector, self).__init__(*args, **kwargs)

    self.value_encoder = None
    self.date_encoder = None
    self.delta_encoder = None
    self.corticalColumn = None
    self.anomalyLikelihood = None
    
    self.modelConfig = None
    self.ccConfig = None
    
    #numpy arrays that hold value/timestamp
    self.value = None
    self.timestamp = None
    self.delta_value = None
    # Keep track of value range for spatial anomaly detection
    self.minVal = None
    self.maxVal = None
    
    self.prevVal = 0

    # Set this to False if you want to get results based on raw scores
    # without using AnomalyLikelihood. This will give worse results, but
    # useful for checking the efficacy of AnomalyLikelihood. You will need
    # to re-optimize the thresholds when running with this setting.
    self.useLikelihood = True
    
    self.tempMem = None
    
    #For hierarchy only
    self.valueCC = None
    self.deltaCC = None
    self.deltaCC2 = None
    self.anomalyCC = None
    
    self.timeCC = None


  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["raw_score","spatial_anomaly"]


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore, rawScore).

    Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
    and "rawScore" corresponds to "anomaly_score". Sorry about that.
    """
    # Get the value
    value = inputData["value"]
    timestamp = inputData["timestamp"]
    
    self.value_encoder.encodeIntoArray(value,self.value)
    self.date_encoder.encodeIntoArray(timestamp,self.timestamp)
    self.delta_encoder.encodeIntoArray(abs(self.prevVal-value), self.delta_value)
    #save value
    self.prevVal = value


    #self.computeHierarchy()
    self.valueCC.computeActiveColumns(np.concatenate((self.timestamp,self.value)))
    self.valueCC.compute(self.delta_value.nonzero()[0])
    # Retrieve the anomaly score and write it to a file
    rawScore = self.valueCC.computeRawAnomalyScore()

    # Update min/max values and check if there is a spatial anomaly
    spatialAnomaly = False
    if self.minVal != self.maxVal:
      tolerance = (self.maxVal - self.minVal) * SPATIAL_TOLERANCE
      maxExpected = self.maxVal + tolerance
      minExpected = self.minVal - tolerance
      if value > maxExpected or value < minExpected:
        spatialAnomaly = True
    if self.maxVal is None or value > self.maxVal:
      self.maxVal = value
    if self.minVal is None or value < self.minVal:
      self.minVal = value

    if self.useLikelihood:
      # Compute log(anomaly likelihood)
      anomalyScore = self.anomalyLikelihood.anomalyProbability(
        inputData["value"], rawScore, inputData["timestamp"])
      logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
      finalScore = logScore
#       if rawScore > 0.5:
#         finalScore = min(logScore * math.exp(0.5*rawScoreDelta),1)
    else:
      finalScore = rawScore

    if self.ccConfig["enableSpatialTrick"]:
      if spatialAnomaly:
        finalScore = 1.0
    spatial_anomaly = 0
    if spatialAnomaly:
      spatial_anomaly = 1

    return (finalScore, rawScore, spatial_anomaly)
  
  def visualizeCC(self):
    cv2.imshow('frame',self.valueCC.visualize())
    if cv2.waitKey(0) & 0xFF== ord('q'):
      return
  
  def computeHierarchy(self):
    self.valueCC.computeActiveColumns(self.value)
    #self.deltaCC.computeActiveColumns(self.delta_value)
    self.timeCC.computeActiveColumns(self.timestamp)

    valueOutput = self.valueCC.getColumnOutput()
    #deltaOutput = self.deltaCC.getColumnOutput()
    timeOutput = self.timeCC.getColumnOutput()        
    
    #self.valueCC.compute(np.concatenate((deltaOutput,timeOutput)).nonzero()[0])
    self.valueCC.compute(timeOutput.nonzero()[0])    
    #self.deltaCC.compute(np.concatenate((valueOutput,timeOutput)).nonzero()[0])
    #self.timeCC.compute(np.concatenate((valueOutput,deltaOutput)).nonzero()[0])
    self.timeCC.compute(valueOutput.nonzero()[0])    
        

  def initialize(self):
    assert self.parameters != None, "We need model parameters to perform feedbackTM..."
    self.modelConfig = self.parameters["modelConfig"]
    self.ccConfig = self.modelConfig["modelParams"]["corticalcolumn"]
    encoderSeed = self.modelConfig["modelParams"]["sensorParams"]["encoders"]["value"]["seed"]
    timeOfDay = self.modelConfig["modelParams"]["sensorParams"]["encoders"]["timestamp_timeOfDay"]["timeOfDay"]
    
    rangePadding = abs(self.inputMax - self.inputMin) * 0.2
    minVal=self.inputMin-rangePadding
    maxVal=self.inputMax+rangePadding
    #print ((maxVal - minVal) /130)
    if self.ccConfig["smartResolution"]:
      resolution = max(0.001,(self.maxValue - self.minValue) / float(self.ccConfig["nrBuckets"]))#max(0.001,(maxVal - minVal) / 130)
    else:
      resolution = max(0.001,(maxVal - minVal) / 130)
    #ENCODERS:
    self.value_encoder = RandomDistributedScalarEncoder(resolution,
                                                        w=21,
                                                        n=400,
                                                        seed=encoderSeed)
    self.delta_encoder = RandomDistributedScalarEncoder(resolution,
                                                        w=21,
                                                        n=400,
                                                        seed=encoderSeed)
    self.date_encoder = DateEncoder(timeOfDay=(timeOfDay[0],timeOfDay[1])) #9.49
    #ENCODER ARRAYS
    self.value = np.zeros(self.value_encoder.getWidth(), dtype='uint32')
    self.timestamp = np.zeros(self.date_encoder.getWidth(), dtype='uint32')
    self.delta_value = np.zeros(self.delta_encoder.getWidth(), dtype='uint32')
    #INITIALIZE Cortical COLUMNS
    self.initializeHierarchy()
    if self.useLikelihood:
      # Initialize the anomaly likelihood object
      numentaLearningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
      #print self.probationaryPeriod-numentaLearningPeriod
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        learningPeriod=numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod-numentaLearningPeriod,
        reestimationPeriod=100
      )
      
  def initializeHierarchy(self):

    neighborCount = 1
    minicolumnCount = 2048
    cellsPerColumnCCTM = 32
    basalWidth = self.delta_encoder.getWidth()#minicolumnCount*cellsPerColumnCCTM*neighborCount

    self.valueCC = CorticalColumn(inputWidth = self.value_encoder.getWidth()+self.date_encoder.getWidth(),
                                  neighborCount = neighborCount,
                                  miniColumnCount = minicolumnCount,
                                  potentialRadius = self.value_encoder.getWidth()+self.date_encoder.getWidth(), #make sure this matches width/2
                                  cellsPerColumnTM = 32,
                                  cellsPerColumnCCTM = cellsPerColumnCCTM,
                                  sparsity = 0.02,
                                  enableLayer4 = True,
                                  enableFeedback = self.ccConfig["enableFeedback"],
                                  burstFeedback = self.ccConfig["burstFeedback"],
                                  delayedFeedback = self.ccConfig["delayedFeedback"],
                                  spSeed = self.modelConfig["modelParams"]["spParams"]["seed"],
                                  tmSeed = self.modelConfig["modelParams"]["tmParams"]["seed"],
                                  SPlearning = True,
                                  basalWidth = basalWidth,
                                  l3SampleSize=self.ccConfig["l3SampleSize"],
                                  l3ActivationThresholdPct=self.ccConfig["l3ActivationThresholdPct"],
                                  l3MinThresholdPct=self.ccConfig["l3MinThresholdPct"],
                                  useApicalModulationBasalThreshold=self.ccConfig["ApicalModulation"],
                                  useApicalTiebreak=self.ccConfig["ApicalTiebreak"],
                                  useIndependentApical=self.ccConfig["IndependentApical"],
                                  useApicalMatch=self.ccConfig["ApicalMatch"],
                                  useTP = False,
                                  reducedBasalPct=self.ccConfig["reducedBasalPct"],
                                  verbosity = 0)
    self.deltaCC = CorticalColumn(inputWidth = self.delta_encoder.getWidth(),
                                  neighborCount = neighborCount,
                                  miniColumnCount = minicolumnCount,
                                  potentialRadius = self.delta_encoder.getWidth(), #make sure this matches width/2
                                  cellsPerColumnTM = 32,
                                  cellsPerColumnCCTM = cellsPerColumnCCTM,
                                  sparsity = 0.02,
                                  enableLayer4 = False,
                                  enableFeedback = self.ccConfig["enableFeedback"],
                                  burstFeedback = self.ccConfig["burstFeedback"],
                                  delayedFeedback = self.ccConfig["delayedFeedback"],
                                  spSeed = self.modelConfig["modelParams"]["spParams"]["seed"],
                                  tmSeed = self.modelConfig["modelParams"]["tmParams"]["seed"],
                                  SPlearning = True,
                                  basalWidth = basalWidth,
                                  l3SampleSize=self.ccConfig["l3SampleSize"],
                                  l3ActivationThresholdPct=self.ccConfig["l3ActivationThresholdPct"],
                                  l3MinThresholdPct=self.ccConfig["l3MinThresholdPct"],
                                  useApicalModulationBasalThreshold=self.ccConfig["ApicalModulation"],
                                  useApicalTiebreak=self.ccConfig["ApicalTiebreak"],
                                  useIndependentApical=self.ccConfig["IndependentApical"],
                                  useApicalMatch=self.ccConfig["ApicalMatch"],
                                  useTP = False,
                                  reducedBasalPct=self.ccConfig["reducedBasalPct"],
                                  verbosity = 0)
    self.timeCC = CorticalColumn(inputWidth = self.date_encoder.getWidth(),
                                 neighborCount = neighborCount,
                                 miniColumnCount = minicolumnCount,
                                 potentialRadius = self.date_encoder.getWidth(), #make sure this matches width/2
                                 cellsPerColumnTM = 32,
                                 cellsPerColumnCCTM = cellsPerColumnCCTM,
                                 sparsity = 0.02,
                                 enableLayer4 = False,
                                 enableFeedback = self.ccConfig["enableFeedback"],
                                 burstFeedback = self.ccConfig["burstFeedback"],
                                 delayedFeedback = self.ccConfig["delayedFeedback"],
                                 spSeed = self.modelConfig["modelParams"]["spParams"]["seed"],
                                 tmSeed = self.modelConfig["modelParams"]["tmParams"]["seed"],
                                 SPlearning = True,
                                 basalWidth = basalWidth,
                                 l3SampleSize=self.ccConfig["l3SampleSize"],
                                 l3ActivationThresholdPct=self.ccConfig["l3ActivationThresholdPct"],
                                 l3MinThresholdPct=self.ccConfig["l3MinThresholdPct"],
                                 useApicalModulationBasalThreshold=self.ccConfig["ApicalModulation"],
                                 useApicalTiebreak=self.ccConfig["ApicalTiebreak"],
                                 useIndependentApical=self.ccConfig["IndependentApical"],
                                 useApicalMatch=self.ccConfig["ApicalMatch"],
                                 useTP = False,
                                 reducedBasalPct=self.ccConfig["reducedBasalPct"],
                                 verbosity = 0)        
  
