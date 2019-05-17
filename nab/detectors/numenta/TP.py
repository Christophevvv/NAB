import numpy as np


class TemporalPooler():
    def __init__(self,
                 columnCount=2048,
                 length=5):
        self.columnCount = 2048
        self.length = length
        self.SDR = np.zeros(self.columnCount,dtype="uint32")

    def compute(self,activeColumns,predictedColumns):
        self._decay()
        self.SDR[predictedColumns] = self.length
        temp = np.zeros(self.columnCount,dtype="uint32")
        #Non-predicted columns only active for 1 time step
        temp[activeColumns] = 1
        self.SDR = np.where(self.SDR == 0, temp, self.SDR)

    def _decay(self):
        #decrement columns that have a
        self.SDR = np.where(self.SDR > 0, self.SDR - 1, self.SDR)


    def getSDR(self):
        return self.SDR
        
        

    
