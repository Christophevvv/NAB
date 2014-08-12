#!/usr/bin/env python
# ----------------------------------------------------------------------
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import os
import yaml

from optparse import OptionParser

import multiprocessing

import lib
from detectors import (NumentaDetector, SkylineDetector)

from collections import defaultdict
import pandas


class Runner(object):

  def __init__(self, options):
    self.options = options
    self.root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    self.config = self.getConfig()
    self.detectors = self.config["AnomalyDetectors"]
    self.resultsDir = os.path.join(self.root, self.config["ResultsDirectory"])
    self.probationaryPercent = self.config["ProbationaryPercent"]
    self.corpusLabel = self.getCorpusLabel()
    self.profiles = self.getProfiles()
    self.numCPUs = self.getNumCPUs()
    self.plot = options.plotResults
    if options.resultsOnly:
      self.results()
    elif options.analysisOnly:
      self.analysis = self.getAnalysis()


  def results(self):
    dataPath = os.path.join(self.root, "data")
    corp = lib.corpus.Corpus(dataPath)

    for detector in self.detectors:

      detectorClassName = lib.util.getDetectorClassName(detector)

      detectorClass = globals()[detectorClassName](
        corpus=corp,
        labels=self.labels,
        name=detector,
        probationaryPercent=self.probationaryPercent,
        outputDir=self.resultsDir,
        numCPUs=self.numCPUs)

      detectorClass.runCorpus()

  def analyzeResults(self):

    for detector in self.detectors:
      resultsDetectorDir = os.path.join(self.resultsDir, detector)
      resultsCorpus = lib.corpus.Corpus(resultsDetectorDir)

      detailedResults = defaultdict(list)

      dataSets = resultsCorpus.getDataSubset('/alerts/')

      for relativePath in dataSets.keys():

        predicted = dataSets[relativePath].data['alert']

        relativePath = lib.util.convertResultsPathToDataPath(relativePath)
        windows = self.corpusLabel.windows[relativePath]
        labels = self.corpusLabel.labels[relativePath]

        # Loop over user profiles
        for profileName, profile in self.profiles.iteritems():
          costMatrix = profile['CostMatrix']

          score = lib.score.Scorer(predicted=predicted, labels=labels, windowLimits=windows, costMatrix=costMatrix)

          costMatrix = score.costMatrix

          detailedResults["File"].extend(relativePath)
          detailedResults["Username"].extend(profileName)
          detailedResults["Score"].extend(score.score)

      detailedResults = pandas.DataFrame(detailedResults)

      detailedResultsPath = os.path.join(resultsDetectorDir, "detailedResults.csv")
      detailedResults.to_csv(detailedResultsPath)

  def getCorpusLabel(self):
    return lib.label.CorpusLabel(options.labelsDir)

  def getConfig(self):
    f = open(os.path.join(self.root, options.config))
    return yaml.load(f)

  def getProfiles(self):
    f = open(os.path.join(self.root, options.profiles))
    return yaml.load(f)

  def getNumCPUs(self):
    if not self.options.numCPUs:
      return multiprocessing.cpu_count()
    return int(self.options.numCPUs)


if __name__ == "__main__":

  parser = OptionParser()
  parser.add_option("-a", "--analyzeOnly",
                    help="Analyze results in the results directory only.",
                    dest="analyzeOnly",
                    default=False,
                    action="store_true")

  parser.add_option("-r", "--resultsOnly",
                    help="Generate detector results but do not analyze results \
                    files.",
                    dest="resultsOnly",
                    default=False,
                    action="store_true")

  parser.add_option("-p", "--plot",
                    help="If you have Plotly installed "
                    "this option will plot results and ROC curves for each \
                    dataset.",
                    dest="plotResults",
                    default=False,
                    action="store_true")

  parser.add_option("--verbosity",
                    default=0,
                    help="Increase the amount and detail of output by setting \
                    this greater than 0.")

  parser.add_option("--config",
                    default="scripts/config/benchmark_config.yaml",
                    help="The configuration file to use while running the "
                    "benchmark.")

  parser.add_option(
                    "--profiles",
                    default="scripts/config/user_profiles.yaml",
                    help="The configuration file to use while running the "
                    "benchmark.")

  parser.add_option("--numCPUs",
                    help="The number of CPUs to use to run the "
                    "benchmark. If not specified all CPUs will be used.")

  parser.add_option("--labelDir",
                    default="labels",
                    help="This holds all the label windows for the corpus.")


  options, args = parser.parse_args()

  Runner(options)