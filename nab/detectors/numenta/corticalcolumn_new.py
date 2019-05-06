import numpy as np
#from nupic.algorithms.spatial_pooler import SpatialPooler
#CPP spatial pooler:
from nupic.bindings.algorithms import SpatialPooler
from htmresearch.algorithms.apical_tiebreak_temporal_memory import ApicalTiebreakPairMemory,ApicalTiebreakSequenceMemory,CrossColumnApicalTiebreakSequenceMemory
#from htmresearch.algorithms.output_layer import OutputLayer
from math import floor
from nupic.algorithms import anomaly

class CorticalColumn():
    def __init__(self,
                 inputWidth,
                 neighborCount=0,
                 miniColumnCount=2048,
                 potentialRadius=16,
                 cellsPerColumnTM=32,
                 cellsPerColumnCCTM=32,#Cross column Temporal memory
                 sparsity=0.02,
                 enableLayer4=True,
                 enableFeedback=True,
                 spSeed = 42,
                 tmSeed = 42,
                 SPlearning=True,
                 verbosity=0):
        self.enableLayer4 = enableLayer4
        self.enableFeedback = enableFeedback
        self.SPlearning = SPlearning
        self.verbosity = verbosity
        self.spOutput = np.zeros(miniColumnCount, dtype="uint32")
        #Active columns:        
        self.activeColumns = np.empty(0, dtype="uint32")
        print "Constructing with inputWidth: " + str(inputWidth)
        self.SP = SpatialPooler(inputDimensions=(inputWidth,),
                                columnDimensions = (miniColumnCount,),
                                potentialRadius = potentialRadius,
                                potentialPct = 0.8,#0.8                                
                                globalInhibition = True,
                                localAreaDensity=-1.0,
                                numActiveColumnsPerInhArea = floor(sparsity*miniColumnCount),
                                stimulusThreshold=0,
                                synPermInactiveDec = 0.0005,
                                synPermActiveInc = 0.003,
                                synPermConnected = 0.2,
                                minPctOverlapDutyCycle=0.001,
                                dutyCyclePeriod=1000,
                                boostStrength = 0.0,
                                seed = spSeed,
                                spVerbosity = 0,
                                wrapAround = True)
        if self.verbosity > 0:
            self.SP.printParameters()

        if self.enableLayer4:
            self.layer4 = Layer4(apicalWidth=miniColumnCount*cellsPerColumnCCTM,
                                 miniColumnCount=miniColumnCount,
                                 cellsPerColumn=cellsPerColumnTM,
                                 sparsity=sparsity,
                                 seed=tmSeed,
                                 verbosity=verbosity)
        self.layer3 = Layer3(neighborCount=neighborCount,
                             basalWidth=54,#neighborCount*miniColumnCount,
                             apicalWidth=0, #No hierarchy yet
                             miniColumnCount=miniColumnCount,
                             cellsPerColumn=cellsPerColumnCCTM,
                             sparsity=sparsity,
                             seed=tmSeed,
                             verbosity=verbosity)
                             
                             
        
    def computeActiveColumns(self,input):
        '''
        Compute the active columns in this cortical column.
        The input is the encoded pixel value.
        This step should be performed before calling compute!
        '''
        assert(len(input) == self.SP.getInputDimensions()[0])
        
        #self.SP.compute(inputVector=input,learn=True,activeArray=self.spOutput)
        self.SP.compute(input,self.SPlearning,self.spOutput)
        #set active column as all nonzero entries of SP output:
        self.activeColumns = self.spOutput.nonzero()[0]

    def compute(self,basalInput=(),apicalInput=()):
        '''
        Make sure to call computeActiveColumns before calling this function.
        This is the general compute function of the cortical column, which calls the
        compute functions of all modeled layers within this cortical column.
        The basal input represent the context input going to the 2/3 layer.
        The basal context of layer4 is internal.
        '''
        #NOTE: we compute layer3 first, this way we can give the active cells (of layer 3)
        #as feedback to layer4 (since the compute of layer4 computes the depolarized
        #cells directly after activating the cells for the current step.
        if self.enableFeedback:
          self.layer3.compute(self.activeColumns,basalInput)
        #layer3 active cells always unique per column?
        if self.enableLayer4:
            if self.enableFeedback:
              self.layer4.compute(self.activeColumns,self.layer3.getWinnerCells())
            else:
              self.layer4.compute(self.activeColumns)

    def computeRawAnomalyScore(self):
        predictedCells = self.layer4.TM.getPredictedCells()
        cellsPerColumn = self.layer4.TM.getCellsPerColumn()   
        prevPredColumns = np.unique(predictedCells/cellsPerColumn)
        return anomaly.computeRawAnomalyScore(self.activeColumns,prevPredColumns)

    def visualize(self):
        '''
        Visualize the cortical column cell activations in each layer.
        '''
        #Create a line to seperate the layers in the output
        line = np.zeros([1,len(self.spOutput),3])
        line[0][:] = (255,255,0)
        #Don't forget this is pair memory (so predictions will seem strange because
        #they happen in the same timestep)
        layer3 = self.layer3.visualize()
        corticalColumnState = np.concatenate((layer3,line))
        if self.enableLayer4:
            layer4 = self.layer4.visualize()        
            corticalColumnState = np.concatenate((corticalColumnState,layer4))
        return corticalColumnState

    def getBasalOutput(self):
        '''
        Return the basal output this column is providing to neighboring column.
        This is simply the output of the spatial pooler (binary array which represent active
        columns)
        '''
        return self.spOutput

    def getColumnOutput(self):
        '''
        Returns the state of the output layer (layer3). This is the output of the column.
        '''
        return self.layer3.getWinnerCellsBinary()

    def getAnomalyScore(self):
        '''
        Return the anomaly score of layer4
        '''
        if self.enableLayer4:
            return self.layer4.getAnomalyScore()
        else:
            return [0,0]
        

    def setSPlearning(self,learn):
        ''' Set whether or not the Spatial Pooler learns '''
        self.SPlearning = learn


class Layer3():
    '''
    Layer3 (or Layer2/3) is the output layer of this cortical column.
    The assumption here is that it gets the same feedforward input as layer 4. But gets
    its context from Layer3 of neighboring cortical columns.
    It provides feedback to layer4.
    '''
    def __init__(self,
                 neighborCount=0,
                 basalWidth=0,
                 apicalWidth=0,
                 miniColumnCount=128,
                 cellsPerColumn=128,#Increase number of possible contexts here
                 sparsity=0.02,
                 seed=42,
                 verbosity=0):
        #SampelSize per neighboring column (number of active cells expected (only one per column?)
        self.ss = 21#floor(miniColumnCount*sparsity)
        #Pair memory so that basal Input is from the same timestep as proximal input
        self.crossColumnTM = ApicalTiebreakPairMemory(columnCount=miniColumnCount,
                                                      basalInputSize=basalWidth,
                                                      apicalInputSize=apicalWidth,
                                                      cellsPerColumn=cellsPerColumn,
                                                      activationThreshold=floor(1*neighborCount*self.ss),
                                                      minThreshold=floor(1*neighborCount*self.ss),
                                                      sampleSize=floor(1*neighborCount*self.ss),
                                                      basalPredictedSegmentDecrement=0.005,
                                                      seed=seed)
        
    def compute(self,proximalInput,basalInput,apicalInput=()):
        '''
        The proximalInput are the active columns of this cortical column.
        The basalInput are the active columns of the neighboring columns concatenated.
        Apical input is still undefined atm.
        '''
        self.crossColumnTM.compute(proximalInput,basalInput)
    

    def visualize(self):
        '''
        Return a visual state of the temporal memory in this layer
        '''
        return self.crossColumnTM.visualize()

    def getActiveCells(self):
        '''
        Return the active cells of the crossColumnTM
        '''
        return self.crossColumnTM.getActiveCells()

    def getWinnerCellsBinary(self):
        '''
        Returns an array where a 1 means a winning cells. This function returns an array rather than just
        the indices.
        '''
        output = np.zeros(self.crossColumnTM.numberOfCells(),dtype ="uint32")
        #Set all active cells to 1
        output[self.getWinnerCells()] = 1
        return output

    def getWinnerCells(self):
        '''
        Return the winner cells of the crossColumnTM
        '''
        return self.crossColumnTM.getWinnerCells()

class Layer4():
    '''
    Layer4 receives the feedforward input. 
    It performs temporal memory with the context coming from other minicolumns in the layer.
    It receives feedback from Layer3.
    '''
    def __init__(self,
                 apicalWidth,
                 miniColumnCount=128,
                 cellsPerColumn=32,
                 sparsity=0.02,
                 seed=42,
                 verbosity=0):
        self.verbosity = verbosity
        #STILL HARDCODED STUFF IN HERE...
        self.sampleSize = floor(miniColumnCount*sparsity) #=2 for standard settings floor(128*0.02)
        self.TM = ApicalTiebreakSequenceMemory(columnCount = miniColumnCount,
                                               #basalInputSize=miniColumnCount*cellsPerColumn,
                                               apicalInputSize=apicalWidth,
                                               cellsPerColumn=cellsPerColumn,
                                               activationThreshold=13,#floor(0.325*self.sampleSize),#3,
                                               reducedBasalThreshold=10,#floor(0.28*self.sampleSize),#13,
                                               initialPermanence=0.21,#0.21
                                               connectedPermanence=0.50,
                                               minThreshold=10,#floor(0.25*self.sampleSize),#10,
                                               sampleSize=20,#floor(0.5*self.sampleSize),#20,
                                               permanenceIncrement=0.1,
                                               permanenceDecrement=0.1,
                                               basalPredictedSegmentDecrement=0.0005,
                                               apicalPredictedSegmentDecrement=0.0005,
                                               maxSynapsesPerSegment=-1,
                                               seed=seed)


    def compute(self,activeColumns,feedback=()):
        ''' 
        Perform layer 4 computation given feedforward input (the active columns)
        and apical feedback 
        '''
        self.TM.compute2(activeColumns=activeColumns,
                         apicalInput=feedback)

    def getAnomalyScore(self):
        '''
        Calculate columnError and cellError:
        columnError = The percentage of columns that was actived but had no predicted cells.
        cellError = The percentage of cells that was activated but was not predicted.
        NOTE: cellError will see more errors than columnError since it is a subset.
        '''
        predictedCells = self.TM.getPredictedCells()
        activeCells = self.TM.getActiveCells()
        cellsPerColumn = self.TM.getCellsPerColumn()
        predictedColumns = np.unique(predictedCells/cellsPerColumn)
        activeColumns = np.unique(activeCells/cellsPerColumn)
        #Percentage of active columns that were not predicted (0%-100%)
        columnError = float(len(np.setdiff1d(activeColumns,predictedColumns)))/len(activeColumns)
        #Percentage of activecells that were not predicted (0%-100%)
        cellError = float(len(np.setdiff1d(activeCells,predictedCells)))/len(activeCells)
        return (columnError,cellError)
        
    def visualize(self):
        '''
        Return a visual state of the temporal memory in this layer
        '''
        return self.TM.visualize()
    
    def reset(self):
        ''' Reset Temporal Memory state '''
        self.TM.reset()

    def getActiveCells(self):
        ''' Get the indices of the active cells within this layer '''
        return self.TM.getActiveCells()

    def getWinnerCells(self):
        ''' 
        Get the indices of the winner cells within this layer.
        These are generally the best matching cells in the column or the correctly predicted cells.
        '''
        return self.TM.getWinnerCells()

    def getBestWinnerCells(self):
        '''
        In case there are multiple winning cells per column this function will select the best winning cell
        per column.
        '''
        return self.TM.getBestWinnerCells()

if __name__ == "__main__":
    cc = CorticalColumn(inputWidth = 2048*32*9)
