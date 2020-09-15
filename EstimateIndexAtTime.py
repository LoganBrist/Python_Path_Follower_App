# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 00:14:09 2018

@author: Logan
"""

           
import numpy as np

           
           
        #Simplified problem: positional data in forms of a line or a circle of length N.
        #Derive movement on the line based on timestamps.
        
        #1. Function is provided with index/time pairs. 
        #2. function assumes it originates from an integer continuous list of indexes, with minimum and maximums being irrelevant.
        #3. If the list is reported as being circular, the max and min index in the timestamp list can be considered the same point.
        #4. Start by finding the rate of change in index at each timestamp needed to traverse the difference in the given time.
        #5. Provided a time, the Function fills in an estimated index on the continuous spectrum.
        
        #ex. given indexes [1,3,5,10] at time stamps [1,2,3,4]. What is the index at time 3.5?
        #   v = (10-5)/1 = 5. index = x0 + v*t = 5 + .5 * 5 = 7.5
        
        #6. To do this, the function first finds the timestamp region the test point 
        #   is in by comparing time.
        #7. Function then uses the velocity in that segment and returns a value.
        
def EstimateIndexAtTime(requestedtime, timestamps, mode = 'v', looping = False, maxindex = 100):
    timelist   = timestamps[0]
    tslength     = len(timelist)-1
    indexlist  = timestamps[1]
    maxtime    = np.amax(timelist)
    mintime    = np.amin(timelist)

    #find timestamp region to use in calculations
    if   requestedtime < mintime:
       return -1
    elif requestedtime >= maxtime:
        region = tslength - 1 #cannot use last region, because velocity calc needs two points
    else:    
        for i in range(0,tslength):
            if requestedtime >= timelist[i] and requestedtime < timelist[i+1]:
                region = i
                break
    
    #region's velocity
    index_0 = indexlist[region]
    index_f = indexlist[region+1]
    t0 =  timelist[region]
    tf =  timelist[region+1]
    dindex = index_f - index_0
    
    
    #The looping path can be treated like a circle of indices. We want the dot
    #to always move towards the next timestamp on the shortest path as to prevent
    # fast backtracking when jumping from high to low or low to high indexes.
    #if the timestamps' indexes are more than halfway around the circle,
    #change the index difference to the shorter path, what's remaining of the circle. 
    #Also negate this value to move in the other direction on the circle.
    
    if looping:
        HalfLoopDistance = maxindex/2
        if abs(dindex) > HalfLoopDistance:
            dindex = -(tslength-dindex) 
    dt = tf - t0
    rateofchange  = dindex / dt
           
    #estimate index for the requested time
    index = int(index_0 + rateofchange * (requestedtime - t0))
    
    #looping past min/max indices
    if looping:
       index = index % maxindex # keeps the index between 0 and max
       
    return index


# an issue with looping is that it uses the highest timestamp index as a maximum..
# When the max index is assumed to overlap with the min index and there are 
# more indexes on the path dot skipping will occur.
#To fix this, the user will need to report the maximum path index (length of the path)
# in addition to the timestamps, so the program can jump to the beggining at the right time.