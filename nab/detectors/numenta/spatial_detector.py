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
#This is the original NAB numenta detector for reference. Since the new one is altered to
#include a model parameter file as input.
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
from nupic.algorithms import anomaly
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood
import numpy as np


# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.005



class SpatialDetector(AnomalyDetector):
  """
  This detector uses an spatial based anomaly detection technique.
  """

  def __init__(self, *args, **kwargs):

    super(SpatialDetector, self).__init__(*args, **kwargs)

    # Keep track of value range for spatial anomaly detection
    self.minVal = None
    self.maxVal = None
    
    #self.upper = self.mean + self.std
    #self.lower = self.mean - self.std
    
#     self.anomalyScore = 0
#     self.anomalyDetector = anomaly.Anomaly(mode='pure')
    #self.s_anomalyLikelihood = AnomalyLikelihood()
    self.s_history = []
    #self.currentAnomalyScore = 0
    #self.useAnomaly = False  
    
    self.s_index = 0
    self.s_historySize = 8640  
    


  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["rawScore"]


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore, rawScore).

    Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
    and "rawScore" corresponds to "anomaly_score". Sorry about that.
    """
    finalScore = 0.0
    # Get the value
    value = inputData["value"]
    if self.s_index < 2:#self.probationaryPeriod:
      self.s_history.append(value)
      self.s_index += 1
    else:
      anomalyProbability = self.computeAnomalyProbability(value)
      finalScore = AnomalyLikelihood.computeLogLikelihood(1-anomalyProbability)
      self.s_history.append(value)
      
    
    
#     finalScore = 0
#     
#     # Update min/max values and check if there is a spatial anomaly
#     spatialAnomaly = False
#     if self.minVal != self.maxVal:
#       tolerance = (self.maxVal - self.minVal) * SPATIAL_TOLERANCE
#       maxExpected = self.maxVal + tolerance
#       minExpected = self.minVal - tolerance
#       if value > maxExpected or value < minExpected:
#         spatialAnomaly = True
#     if self.maxVal is None or value > self.maxVal:
#       self.maxVal = value
#     if self.minVal is None or value < self.minVal:
#       self.minVal = value
#  
#     if spatialAnomaly:
# #       if value > self.upper or value < self.lower:      
#       finalScore = 1.0
# #     if value > self.upper:
# #       finalScore = 1.0
# #     if value < self.lower:
# #       finalScore = 1.0
# #     if value > self.maxValue:
# #       finalScore = 1.0
# #     elif value < self.minValue:
# #       finalScore = 1.0
# #     else:
# #       finalScore = 0
#     valInput = value - self.inputMin
#     valInput = round(float(valInput) / self.valRange,1)
    anomalyScore = self.anomalyLikelihood.anomalyProbability(
      value, finalScore, inputData["timestamp"])
    logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
    finalScore = logScore

    return (finalScore,0)


  def initialize(self):
    # Initialize the anomaly likelihood object
    self.numentaLearningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
    self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        learningPeriod=self.numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod-self.numentaLearningPeriod,
        reestimationPeriod=100
    )
  
  def computeAnomalyProbability(self,input):
      values = np.asarray(self.s_history)
      distributionParams = {}
      window = values[-self.s_historySize:]
      distributionParams["mean"] = np.mean(window)
      distributionParams["stdev"] = np.std(window)
      return self.tailProbability(input,distributionParams)

  def tailProbability(self, x, distributionParams):
      """
      Given the normal distribution specified by the mean and standard deviation
      in distributionParams, return the probability of getting samples further
      from the mean. For values above the mean, this is the probability of getting
      samples > x and for values below the mean, the probability of getting
      samples < x. This is the Q-function: the tail probability of the normal distribution.

      :param distributionParams: dict with 'mean' and 'stdev' of the distribution
      """
      if "mean" not in distributionParams or "stdev" not in distributionParams:
          raise RuntimeError("Insufficient parameters to specify the distribution.")

      if x < distributionParams["mean"]:
          # Gaussian is symmetrical around mean, so flip to get the tail probability
          xp = 2 * distributionParams["mean"] - x
          return self.tailProbability(xp, distributionParams)
    
      # Calculate the Q function with the complementary error function, explained
      # here: http://www.gaussianwaves.com/2012/07/q-function-and-error-functions
      if distributionParams["stdev"] == 0:
        distributionParams["stdev"] = 1
      z = (x - distributionParams["mean"]) / distributionParams["stdev"]
      return 0.5 * math.erfc(z/1.4142)    
  

