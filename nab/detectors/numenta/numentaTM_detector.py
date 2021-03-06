# ----------------------------------------------------------------------
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
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

import os
import math
import simplejson as json

from nupic.algorithms import anomaly_likelihood
from nupic.frameworks.opf.common_models.cluster_params import (
  getScalarMetricWithTimeOfDayAnomalyParams)
try:
  from nupic.frameworks.opf.model_factory import ModelFactory
except:
  # Try importing it the old way (version < 0.7.0.dev0)
  from nupic.frameworks.opf.modelfactory import ModelFactory

from nab.detectors.numenta.numenta_detector import NumentaDetector
from nab.detectors.context_ose.cad_ose import ContextualAnomalyDetectorOSE



class NumentaTMDetector(NumentaDetector):
  """
  This detector uses the implementation of temporal memory in
  https://github.com/numenta/nupic.core/blob/master/src/nupic/algorithms/TemporalMemory.hpp.
  It differs from its parent detector in temporal memory and its parameters.
  """

  def __init__(self, *args, **kwargs):

    super(NumentaTMDetector, self).__init__(*args, **kwargs)


  def initialize(self):
    rangePadding = abs(self.inputMax - self.inputMin) * 0.2
    if not (self.parameters == None):
      print "parameters found!"
      self.genericConfig = self.parameters["generic"]
      minVal=self.inputMin-rangePadding
      maxVal=self.inputMax+rangePadding
      # Handle the corner case where the incoming min and max are the same
      if minVal == maxVal:
        maxVal = minVal + 1  
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
    
    
    
#     
#     if not (self.parameters == None):
#       self.genericConfig = self.parameters["generic"]
#     # Get config params, setting the RDSE resolution
#     rangePadding = abs(self.inputMax - self.inputMin) * 0.2
# 
#     modelParams = getScalarMetricWithTimeOfDayAnomalyParams(
#       metricData=[0],
#       minVal=self.inputMin-rangePadding,
#       maxVal=self.inputMax+rangePadding,
#       minResolution=0.001,
#       tmImplementation="tm_cpp"
#     )["modelConfig"]
# 
#     self._setupEncoderParams(
#       modelParams["modelParams"]["sensorParams"]["encoders"])
# 
#     self.model = ModelFactory.create(modelParams)
# 
#     self.model.enableInference({"predictedField": "value"})
# 
#     # Initialize the anomaly likelihood object
#     numentaLearningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
#     self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
#       learningPeriod=numentaLearningPeriod,
#       estimationSamples=self.probationaryPeriod-numentaLearningPeriod,
#       reestimationPeriod=100
#     )
