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

from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.common_models.cluster_params import (
  getScalarMetricWithTimeOfDayAnomalyParams)
try:
  from nupic.frameworks.opf.model_factory import ModelFactory
except:
  # Try importing it the old way (version < 0.7.0.dev0)
  from nupic.frameworks.opf.modelfactory import ModelFactory

from nab.detectors.base import AnomalyDetector
from nab.detectors.context_ose.cad_ose import ContextualAnomalyDetectorOSE
from nab.detectors.numenta.spatial_detector import SpatialDetector
import numpy as np
import copy

# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.05



class NumentaDetector(AnomalyDetector):
  """
  This detector uses an HTM based anomaly detection technique.
  """

  def __init__(self, *args, **kwargs):

    super(NumentaDetector, self).__init__(*args, **kwargs)

    self.model = None
    self.modelConfig = None
    self.sensorParams = None
    self.anomalyLikelihood = None
    # Keep track of value range for spatial anomaly detection
    self.minVal = None
    self.maxVal = None

    # Set this to False if you want to get results based on raw scores
    # without using AnomalyLikelihood. This will give worse results, but
    # useful for checking the efficacy of AnomalyLikelihood. You will need
    # to re-optimize the thresholds when running with this setting.
    self.useLikelihood = True
    
    self.stepsize = None
    self.prevTimestamp = None
    self.genericConfig = None
    self.dataIndex = 0
    self.dataWindows = []
    self.numentaLearningPeriod = None
    
    self.cadose = None
    self.relativePath = None
    self.nrResets = 0
    #self.spatialDetector = SpatialDetector(*args,**kwargs)
    #self.spatialDetector.initialize()
    #print self.parameters["generic"]["doubleTM"]
    #print self.dataSet.data.shape
    if self.parameters["generic"]["doubleTM"]:
      #print self.parameters["generic"]["doubleTM"]
      parameters2 = copy.deepcopy(self.parameters)
      parameters2["generic"]["doubleTM"] = False
      #print parameters2
      parameters2["generic"]["smartResolution"] = 1
      parameters2["generic"]["nrBuckets"] = 130
      self.TM2 = NumentaDetector(copy.deepcopy(self.dataSet),
                                 copy.deepcopy(self.probationaryPercent),
                                 parameters2)
      #self.TM2.initialize()
      #print "initialized"


  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["raw_score","spatial_anomaly"]


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore, rawScore).

    Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
    and "rawScore" corresponds to "anomaly_score". Sorry about that.
    """
    #return (0,0,0)
    # Send it to Numenta detector and get back the results
    reset = False
    if self.genericConfig["missingValues"]:
      timestamp = inputData["timestamp"]
      if self.prevTimestamp == None:
        self.prevTimestamp = timestamp
      else:
        duration = timestamp - self.prevTimestamp
        if self.dataIndex < self.numentaLearningPeriod:
          self.dataWindows.append(duration.total_seconds())
        elif self.dataIndex == self.numentaLearningPeriod:
          arr = np.asarray(self.dataWindows)
          arr.sort()          
          self.stepsize = np.median(arr)
          print "Settled on stepsize"
          print self.stepsize
#           f = open("stepsize_time.txt","a+")
#           f.write(str(self.relativePath) + ": " + str(self.stepsize) + "\n")
#           f.close()
#         if self.stepsize == None:
#           self.stepsize = duration.total_seconds()
        else:
#           if duration.total_seconds() != self.stepsize:
#             #print duration.total_seconds()
#             #print self.stepsize
#             self.stepsize = duration.total_seconds()
#             self.model.resetSequenceStates()
#             #print "RESETTING"
#           if duration.total_seconds() < self.stepsize:
#             self.stepsize = duration.total_seconds()
          if duration.total_seconds() > self.genericConfig["missingThreshold"]*self.stepsize:
            #print duration.total_seconds()
            #print self.stepsize
            #WE missed a value, reset TM
            #print "RESETTING"
            reset = True
            self.model.resetSequenceStates()
            self.nrResets += 1
          #else:
            #print "EQUAL"
      self.prevTimestamp = timestamp
      self.dataIndex += 1
    #return (0,0,0)   
      
    
    result = self.model.run(inputData)

    # Get the value
    value = inputData["value"]

    # Retrieve the anomaly score and write it to a file
    rawScore = result.inferences["anomalyScore"]
    #dont want increase here
    if reset:
      rawScore = 0.0

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
      if reset and self.genericConfig["ignoreReset"]:
        finalScore = 0.0
      else:
        # Compute log(anomaly likelihood)
        anomalyScore = self.anomalyLikelihood.anomalyProbability(
          inputData["value"], rawScore, inputData["timestamp"])
        logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
        finalScore = logScore
    else:
      finalScore = rawScore

    if self.genericConfig["OSE"]:
      anomalyScore = self.cadose.getAnomalyScore(inputData)
      #finalScore = anomalyScore
      if self.genericConfig["rescaleOSE"]:
        anomalyScore = anomalyScore * (2.0/3)
      finalScore = max(finalScore,anomalyScore)
      
    if self.genericConfig["SPATIAL"]:
      anomalyScore = self.spatialDetector.handleRecord(inputData)[0]
      finalScore = max(finalScore,anomalyScore)
      
    if self.parameters["generic"]["doubleTM"]:
      #print "perform"
      score = self.TM2.handleRecord(inputData)[0]
      #print "back"
      finalScore = max(finalScore,score)
      
    if self.genericConfig["enableSpatialTrick"]:
      if spatialAnomaly:
        finalScore = 1.0
    spatial_anomaly = 0
    if spatialAnomaly:
      spatial_anomaly = 1

    return (finalScore, rawScore, spatial_anomaly)


  def initialize(self):
    rangePadding = abs(self.inputMax - self.inputMin) * 0.2
    if not (self.parameters == None):

      self.genericConfig = self.parameters["generic"]
      minVal=self.inputMin-rangePadding
      maxVal=self.inputMax+rangePadding
      # Handle the corner case where the incoming min and max are the same
      if minVal == maxVal:
        maxVal = minVal + 1  
      if self.minValue == self.maxValue:
        self.maxValue = self.minValue + 1          
      valueEncoder = self.parameters["modelConfig"]["modelParams"]["sensorParams"]["encoders"]["value"]
      type = valueEncoder["type"]
      if self.genericConfig["smartResolution"]:

        resolution = max(0.001,(self.maxValue - self.minValue) / float(self.genericConfig["nrBuckets"]))
      else:
        resolution = max(0.001,(maxVal - minVal) / float(self.genericConfig["nrBuckets"]))
      self.modelConfig = self.parameters["modelConfig"]
      if self.genericConfig["valueOnly"]:
        self.modelConfig["modelParams"]["sensorParams"]["encoders"]["timestamp_timeOfDay"] = None
      valueEncoder["resolution"] = resolution
      if type == "ScalarEncoder":
        valueEncoder["n"] = 0
        minVal = self.inputMin
        maxVal = self.inputMax
        if minVal == maxVal:
          maxVal = minVal + 1
        valueEncoder["minval"] = minVal
        valueEncoder["maxval"] = maxVal
        if "seed" in valueEncoder:
          valueEncoder.pop("seed")
      if type == "AdaptiveScalarEncoder":
        if self.genericConfig["adaptiveMinMax"]:
          valueEncoder["minval"] = minVal
          valueEncoder["maxval"] = maxVal
        valueEncoder.pop("resolution")
        if "seed" in valueEncoder:
          valueEncoder.pop("seed")
          
      if self.genericConfig["smartTime"]:
        self.modelConfig["modelParams"]["sensorParams"]["encoders"]["timestamp_timeOfDay"]["timeOfDay"] = [21,self.genericConfig["radius"]*self.radius]
        print self.modelConfig["modelParams"]["sensorParams"]["encoders"]["timestamp_timeOfDay"] 
      #RESOLUTION STUFF
#       f = open("resolution.txt","a+")
#       range_dumb = round(maxVal-minVal,2)`
#       range_smart = round(self.maxValue - self.minValue,2)
#       res_dumb = round(max(0.001,(maxVal - minVal) / float(self.genericConfig["nrBuckets"])),2)
#       res_smart = round(max(0.001,(self.maxValue - self.minValue) / float(self.genericConfig["nrBuckets"])),2)
#       f.write(str(self.relativePath) + "," + str(round(minVal,2)) + "," + str(round(maxVal,2)) + "," + str(range_dumb) 
#               + "," + str(res_dumb) + "," + str(round(self.minValue,2)) + "," + str(round(self.maxValue,2)) + "," 
#               + str(range_smart) + "," + str(res_smart) + "," + str(round(float(range_dumb)/range_smart,2)) + "\n")
#       f.close()
      #END RESOLUTION STUFF
      #print valueEncoder
      self.sensorParams = valueEncoder #To check if equal to _setupEncoderParams assignment
    else:
      self.modelConfig = getScalarMetricWithTimeOfDayAnomalyParams(
        metricData=[0],
        minVal=self.inputMin-rangePadding,
        maxVal=self.inputMax+rangePadding,
        minResolution=0.001,
        tmImplementation = "cpp"
        )["modelConfig"]
      self._setupEncoderParams(self.modelConfig["modelParams"]["sensorParams"]["encoders"])

    #print self.sensorParams
    self.model = ModelFactory.create(self.modelConfig)

    self.model.enableInference({"predictedField": "value"})

    if self.useLikelihood:
      # Initialize the anomaly likelihood object
      self.numentaLearningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        learningPeriod=self.numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod-self.numentaLearningPeriod,
        reestimationPeriod=self.genericConfig["reestimationPeriod"]
      )
    
    if self.genericConfig["OSE"]:
        self.cadose = ContextualAnomalyDetectorOSE (
          minValue = self.inputMin,
          maxValue = self.inputMax,
          restPeriod = self.probationaryPeriod / 5.0,
          )
    if self.parameters["generic"]["doubleTM"]:
      self.TM2.initialize()  


  def _setupEncoderParams(self, encoderParams):
    # The encoder must expect the NAB-specific datafile headers
    encoderParams["timestamp_dayOfWeek"] = encoderParams.pop("c0_dayOfWeek")
    encoderParams["timestamp_timeOfDay"] = encoderParams.pop("c0_timeOfDay")
    encoderParams["timestamp_timeOfDay"]["fieldname"] = "timestamp"
    encoderParams["timestamp_timeOfDay"]["name"] = "timestamp"
    encoderParams["timestamp_weekend"] = encoderParams.pop("c0_weekend")
    encoderParams["value"] = encoderParams.pop("c1")
    encoderParams["value"]["fieldname"] = "value"
    encoderParams["value"]["name"] = "value"
 
    self.sensorParams = encoderParams["value"]
    
  def setPath(self,path):
    self.relativePath = path
    
  def writeResets(self):
    f = open("resets.txt","a+")
    f.write(str(self.relativePath)+ " & " + str(self.nrResets) + "\n")
    f.close()
