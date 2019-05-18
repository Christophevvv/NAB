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
    
    self.upper = self.mean + self.std
    self.lower = self.mean - self.std
    


  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["rawScore"]


  def handleRecord(self, inputData):
    """Returns a tuple (anomalyScore, rawScore).

    Internally to NuPIC "anomalyScore" corresponds to "likelihood_score"
    and "rawScore" corresponds to "anomaly_score". Sorry about that.
    """

    # Get the value
    value = inputData["value"]
    
    
    finalScore = 0
    
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
 
    if spatialAnomaly:
#       if value > self.upper or value < self.lower:      
      finalScore = 1.0
#     if value > self.upper:
#       finalScore = 1.0
#     if value < self.lower:
#       finalScore = 1.0
#     if value > self.maxValue:
#       finalScore = 1.0
#     elif value < self.minValue:
#       finalScore = 1.0
#     else:
#       finalScore = 0

    return (finalScore,0)


  def initialize(self):
    pass

