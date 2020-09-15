import cv2
import numpy as np
import time as tm
import datetime



'''
##############################################################################
#Path Finding class
##############################################################################
'''
class PathApp:

    def __init__(self, imgname, cleanimg, pathcolor, autogenerate = True, loop = False, mode = 'v'):
        self.imgname       = imgname
        self.cleanimg      = cleanimg
        self.img           = cv2.imread(imgname)
        self.background    = self.img.copy()
        self.pathcolor     = pathcolor
        self.loop          = loop
        self.width,self.height,self.layers = self.img.shape
        
        self.colordistance  = None
        self.skelaton       = None
        self.unorderedpath  = None
        self.path           = None
        self.startingpoints = None
        self.timestamps     = []
        self.movement       = None
        self.fps            = 60
     
          
        if autogenerate == True:
            self.ColorDistanceFilter(mode = 'BW', threshold = 220)   #creates colordistance
            self.skelatonize()           #creates skelaton, if no BW input assumes 127/255 split
            self.CoordinateFilter()                       #Creates unorderedpath
            self.LonePointFilter(ROI = 8, n_min = 7)      #Edits unorderedpath
            self.NeighborFilter(ROI = 10, preview = True) #Creates startingpoints
            self.createpath()                             #Creates path
            self.TimeStamps(mode = "typing")              #Creates timestamps
            self.FormContinuousPath(mode)           #creates movement
            if input("Save to video? (y/n)") == 'y':
               self.savevideo(background = cleanimg ,savename = 'video.avi')            



     
        
    def draw(self,img):
        cv2.imshow('fghj',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def drawpoints(self,points,thickness = 5, gradient = True):
        img = self.img.copy()
        inc = 255/ len(points)
        color = 0
        
        for p in points:
           cv2.circle(img, (p[1],p[0]), thickness, color, -1)
           color += inc

        cv2.imshow('fghj',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    '''
    ##############################################################################
    #Path Finding Functions
    ##############################################################################
    '''
    #########################################################################
    def ColorDistanceFilter(self, mode = 'BW', threshold = 127): 
    #########################################################################    
    #img       = RGB image  (x,y,3)
    #color     = RGB color [r,g,b] to reference for calculating distance    
    
    # Takes in an RGB image and a reference color. Returns a grayscale image where 
    # 0 is complete color matching and 255 is no color matching.    
    
    # Returns: Grayscale image (x,y)  
    #########################################################################
       print("1. Starting ColorDistanceFilter...")
       img   = self.img     
       color = self.pathcolor
       width, height, layers = img.shape
       distanceimage   = np.zeros((width,height))
       
       for i in range (0,width): 
          for j in range (0,height):
             bdif = img[i,j][0] - color[0] 
             gdif = img[i,j][1] - color[1] 
             rdif = img[i,j][2] - color[2]
             d  = np.sqrt(bdif**2 + gdif**2 + rdif**2)        #color distance
             distanceimage[i][j]  = 255 - int(d * 255 / 441)  #convert to grayscale
             #442 is max difference, sqrt((255^2)+(255^2)+(255^2)) = 441.7
             
             #converts to BW
             if   (mode == 'BW') and (distanceimage[i][j] >= threshold):
                 distanceimage[i][j]  = 255
             elif (mode == 'BW'):
                 distanceimage[i][j] = 0
             else:
                 pass

       self.colordistance = None          
       self.colordistance = distanceimage
       print("Color distance image saved as obj.colordistance.\n")
    
    def skelatonize(self): 
    #########################################################################    
    #img       = Black and White image (x,y)  , if not assumed 127/255 split
    
    # Takes in a BW image that has a path drawn on it, and shrinks the path to be
    # one pixel wide.     
    
    # Returns: Black and White Image (x,y)
    # SOURCE: opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html    
    #########################################################################     
        print("2. Starting skelatonize...")
        imgc = np.uint8(self.colordistance.copy())
        size = np.size(imgc)
        skel = np.zeros(imgc.shape,np.uint8)
         
        ret,imgc = cv2.threshold(imgc,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False
         
        while( not done):
            eroded = cv2.erode(imgc,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(imgc,temp)
            skel = cv2.bitwise_or(skel,temp)
            imgc = eroded.copy()
         
            zeros = size - cv2.countNonZero(imgc)
            if zeros==size:
                done = True
                
        self.skelaton = skel 
        self.skelaton = skel
        print("Skelaton generated and saved as obj.skelaton.\n")

    
    #########################################################################
    def CoordinateFilter(self): 
    #########################################################################    
    #img       = BW image (x,y)
    #threshold = 8-bit grayscale number to compare coordinates to (scalar int)
    #side      = 'under','over', or 'equal'. Says which region the compared coordinate
    #            needs to fall in to be added to the return list. Defaults to 'over'.  
     
    # Takes in a gray scale image and returns all coordinates that are (under/over/equal) to
    # the threshold limit.     
    
    # Returns: List of coordinates (size 2 tuples)
    #########################################################################
        print("3. Starting CoordinateFilter...")
        img  = self.skelaton 
        path = []  
        width, height = img.shape
        for i in range (0,width): 
          for j in range (0,height):
             point = (i,j)
             val   = img[i][j]
             
             if   val == 255:
                    path.append(point)
        self.unorderedpath = None
        self.unorderedpath = path            
        print("Skelaton path coordinates saved as obj.unorderedpath.\n")

    #########################################################################
    def LonePointFilter(self, ROI = 8, n_min = 7):
    #########################################################################    
    #coordlist = list of coordinates (size 2 tuples)
    #ROI       = Distance to consider a coordinate a neighbor (scalar int)
    #n_min     = Minimum number of neighbors needed to not remove coordinate (scalar int)   
        
    # LonePointFilter takes in a list of image coordinates and removes coordinates
    # that have too few neighboring pixels. This decreases the likelihood of false
    # positive points being added to the path. 
        
    # Returns: List of coordinates (size 2 tuples)
    #########################################################################    
        print("4. Starting LonePointFilter...")
        coordlist_new = self.unorderedpath.copy()
        cnt = 0
        for p1 in coordlist_new:
            n = 0
            #count neighbors
            for p2 in coordlist_new:
                v = np.subtract(p1,p2)
                m = np.linalg.norm(v)
                if m < ROI:   
                    n += 1
            if n < n_min:
               coordlist_new.remove(p1)
               cnt += 1
        
        self.unorderedpath = None 
        self.unorderedpath = coordlist_new      
        print("Lone point filter removed " + str(cnt) + " points.\n")

        
    #########################################################################   
    def startingpointprompt(self):
    #########################################################################
    #
    #########################################################################    
        c  = self.img.copy()
        sp = self.startingpoints.copy()
        
        print('Possible starting points:')
        for i in range(len(sp)):
            p = (sp[i][1],sp[i][0])
            cv2.circle(c, p, 5, (0,0 ,0), -1)
            print(p)
        cv2.imshow('Autogenerated end points for path',c)
        print('Autogenerated endpoints are shown. Click on the image and hit Enter to continue.')    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('with ROI: ' + str(self.ROI_n))
        print("""Recalculate with new ROI(n), choose starting point manually
              (m), or are marked points correct(c)?""")
        inp = input(': ')
        
        return inp
        
    #########################################################################   
    def choosepoint(self):
    #########################################################################    
         coordlist    = self.unorderedpath.copy()
         img          = self.img.copy()
         point        = (0,0)
         self.mouse_x      = 0
         self.mouse_y      = 0
         flgs         = [False,False,False]
         
         def mousefunctions(event,x,y,flags,param):
            self.mouse_x, self.mouse_y = x,y
            if flags ==  cv2.EVENT_FLAG_LBUTTON:
                flgs[0] = True
            elif flags ==  cv2.EVENT_FLAG_RBUTTON:
                flgs[1] = True   
            elif flags ==  cv2.EVENT_FLAG_SHIFTKEY: 
                flgs[2] = True
        
             
         cv2.imshow("Click path end/start point",img)
         cv2.setMouseCallback("Click path end/start point", mousefunctions) 
         cv2.waitKey(1)
         print("Left click to select, right click to continue")
         
         while True:
            if flgs[0] == True:
                p     = (self.mouse_y,self.mouse_x) 
                point = self.findclosestpoint(coordlist,p)
                img = self.img.copy()
                cv2.circle(img, (point[0][1],point[0][0]), 5, (0,0 ,0), -1)

            if flgs[1] == True:
               print('Point ' + str(point) + ' selected.') 
               break
           
            flgs = [False,False,False]
            cv2.imshow("Click path end/start point",img)
            cv2.waitKey(1)
            
         cv2.destroyAllWindows()  
         self.startingpoints = None
         self.startingpoints = point
     
    #########################################################################              
    def findclosestpoint(self,coordlist,p2):
    #########################################################################   
        min_m = np.uint64(-1)
        closestpoints = []
        for p1 in coordlist:
                v = np.subtract((p1),(p2))
                m = np.linalg.norm(v)
                
                if (m == min_m) and (p1 != p2):
                    closestpoints.append(p1)
                    
                elif (m < min_m) and (p1 != p2):
                    closestpoints = []
                    closestpoints.append(p1)
                    min_m = m
                   
        return closestpoints
    
    # I turned closestpoints into a list
    # I made the function check for equally spaced neighbors. It returns all
    # nearest points, instead of just one. It will be left up to the create path 
    # function to eliminate all points in the returned list.
    
    # Previously, the dot would double back on its path because the path was wide
    # in some instances. This should prevent that.
    #########################################################################
    def NeighborFilter(self,ROI = 10, preview = False):
    #########################################################################    
    #coordlist = list of coordinates (size 2 tuples)
    #ROI       = Distance to consider a coordinate a neighbor (scalar int)
        
    # Takes in a list of coordinates and returns the coordinate(s) with the fewest 
    # neighbors in the range of ROI pixels. The list represents possible end-points 
    # of a path. If there are many, that means the user drew more than one path or 
    # the path coordinates weren't filtered well enough. The user would need to select
    # an appropriate starting point from the list. If there are two coordinates returned,
    # a single path is detected and either can be chosen as starting points. If there is no 
    # point, it is possible that a loop was drawn, and any starting point can be chosen.
    
    # Can use a return flag, indicating what method above the user should use.    
        
    # Returns: List of coordinates (size 2 tuples)
    #########################################################################  
        print("5. Starting endpoint filter (NeighborFilter)...")
        if self.loop == True:
            self.startingpoints = self.unorderedpath[0]
            print("Path is a loop, starting point is not needed.\n")
            return
        
        self.ROI_n = ROI  #makes local variable ROI accessible
        coordlist = self.unorderedpath.copy()
        n_min     = 255
        endpoints = []
        
        for p1 in coordlist:
            n = 0
            for p2 in coordlist:
                v = np.subtract(p1,p2)
                m = np.linalg.norm(v)
                if m < ROI:   #if test point is in region, increment neighbor count
                    n += 1
            if n < n_min:
               endpoints.clear()
               endpoints.append(p1) 
               n_min = n
            elif n == n_min:
               endpoints.append(p1)
               
        self.startingpoints = endpoints      
        print("Starting point list generated and saved as obj.startingpoints.")

        if preview == True: 
            inp = self.startingpointprompt()
            
            if   inp == 'n':
                user_ROI = int(input('New ROI: '))
                self.NeighborFilter(ROI = user_ROI, preview = True)
            elif inp == 'm':
                self.choosepoint()
            elif inp == 'c':
                print('Point chosen')
            else:
                print('invalid input')
                self.startingpointprompt()
              
                
    
    #########################################################################
    def createpath(self):
    #########################################################################    
    #coordlist     = list of coordinates (size 2 tuples)
    #Startingpoint = end point on a path to start algorithm (size 2 tuple)
        
    # Takes in a list of coordinates and returns the same list, sorted such that 
    # the coordinates form a path starting at the starting point. This is done by 
    # iteratively finding the closest coordinate to the reference coordinate, and 
    # then making that the new reference coordinate and adding it to the returned list.
         
    # Returns: List of coordinates (size 2 tuples)
    #########################################################################     
        print("6. Starting createpath...")
        remainingpoints = self.unorderedpath.copy()
        p1              = self.startingpoints[0]
        sortedpath      = []
        while len(remainingpoints) > 0:           
            closestpoints = self.findclosestpoint(remainingpoints,p1)
            for point in closestpoints:    
                remainingpoints.remove(point) 
            sortedpath.append(closestpoints[0])
            p1 = closestpoints[0]
            
        self.path = sortedpath
        print("Path ordered and saved as obj.path.\n")
   
    '''
    ##############################################################################
    #User Drawing Functions
    ##############################################################################
    '''
  
    #########################################################################
    def markpoint(self,image,time,x,y):
    #########################################################################    
    #
     
    # Returns: 
    #########################################################################  
        #find closest coordinate to mouse point x,y
        p1           = (y,x)  
        cp           = (0,0) #closest point
        min_m        = np.uint64(-1)
        img          = image
        coordinates  = self.path.copy()
        
        for p2 in coordinates:
            v = np.subtract(p1,p2)
            m = np.linalg.norm(v)
            if m <= min_m:
                cp = p2
                min_m        = m
        indx = coordinates.index(cp)
        
        #draw point  
        cv2.circle(img,(cp[1],cp[0]), 10, (0,0,0), -1)
        #request time(if applicable)
        if time == None:
            time = int(input("enter time (seconds): "));
        #save point
        self.timestamps.append([time,indx])
        #print point
        print('%.2f' % time,indx)        
          
    #########################################################################        
    def TimeStamps(self,mode = "typing"):
    #########################################################################    
    #coordlist  =  list of coordinates (size 2 tuples)
    
    # Starts a live timer and has user click the path coordinate on the original 
    # image for where they want an animated object to be at that instance in time.     
        
    #Returns: list of (timestamp,index of coordlist), to be used as (time,pos) points
    #         in motion calculations along the coordlist path.    
    ######################################################################### 
        
        flgs = [False,False,False]
        self.timestamps = []
        def mousefunctions(event,x,y,flags,param):
            self.mouse_x,self.mouse_y = x,y 
            if flags ==  cv2.EVENT_FLAG_LBUTTON:
                flgs[0] = True
            elif flags ==  cv2.EVENT_FLAG_RBUTTON:
                flgs[1] = True        
            elif flags ==  cv2.EVENT_FLAG_SHIFTKEY: 
                flgs[2] = True

        img        = self.img.copy() 
        cv2.imshow('image', img) 
        cv2.setMouseCallback("image", mousefunctions) 
        pauseflag  = False
        time       = 0
        t0         = tm.time()
#-----------------------------------------------------------------------------        
        if mode == "clicking":
            print('Enter when ready to start timer.')
            print('Right click to end timer.')
            print('Shift click to pause/resume timer.')
    
            while True:
                if pauseflag == True:
                    #button events
                    if   flgs[0] == True:
                         pass
                    elif flgs[1] == True:
                         break
                         print('stop')
                    elif flgs[2] == True:
                         pauseflag = False
                         print('resume')     
                    flgs = [False,False,False] #reset flags 
        
                elif pauseflag == False:           
                    #calculate time
                    t = tm.time()
                    if t - t0 >= .01:
                        t0 = t
                        time += .01   
                        print(str(datetime.timedelta(seconds = int(time))))  
                    
                    #button events
                    if   flgs[0] == True:
                         self.markpoint(img,time,self.mouse_x,self.mouse_y) #draws and save to timestamps
                    elif flgs[1] == True:
                         pauseflag = True
                         print('pause')
                    elif flgs[2] == True:
                         break
        
                    flgs = [False,False,False] #reset flags
                
                cv2.imshow('image', img) 
                cv2.waitKey(1) 
            
            cv2.destroyAllWindows() 
#-----------------------------------------------------------------------------    
        if mode == "typing":
            print("Left click to mark points.")
            print("Right click to stop marking phase and to begin adding timestamps.")
            
            while True:
                #button events
                if   flgs[0] == True:
                     self.markpoint(img,time,self.mouse_x,self.mouse_y) #draws and save to timestamps
                elif flgs[1] == True:
                     break
                flgs = [False,False,False] #reset flags
                cv2.imshow('image', img) 
                cv2.waitKey(1)
                
            #edit list time stamps
            markedpoints = self.timestamps

            for point in markedpoints:  
              time = point[0]
              indx = point[1]
              y    = self.path[indx][1]
              x    = self.path[indx][0]
              img  = self.img.copy()
              cv2.circle(img,(y,x), 10, (0,0,0), -1)
              cv2.imshow('image', img) 
              cv2.waitKey(1)

              time = float(input("Time for this dot position (seconds): ")) 
              self.timestamps[markedpoints.index(point)] = [time,indx]
                   
        print('stop')        
        cv2.destroyAllWindows()        
        
    '''
    ##############################################################################
    #Path calculation and video exporting
    ##############################################################################
    '''    
    #########################################################################        
    def FormContinuousPath(self, mode = 'v'):
    #########################################################################    
    #coordlist  =  list of coordinates (size 2 tuples)
    #Returns tuple (time,coordinate) continuous   
    
    #function needs to take in the ordered list of points that makes up the drawn path,
    #and also take in the timestamps produces by the user.
    #function returns a list of N (fps * video time) points that the dot is to be located
    #at at each frame.
    ######################################################################### 
       idxlist = self.timestamps  #holds index values of path coordinates and the times
                                  #the dot is located at each.
       path    = self.path.copy() #holds list of path coordinates 
       
 #############################################################################              
    #constant velocity mode 
       if mode == 'v':
            t =  [p[0] for p in idxlist]  
            videolength   = t[len(t)-1] #seconds
            fps           = 60 #fps seconds 
            framecount = int(fps * videolength)
            tf   = []
            indx = []
            for i in range (0,framecount):
                t = i/fps
                tf.append(t)
                indx.append(EstimateIndexAtTime(t, self.timestamps, mode = 'v', looping = self.loop, maxindex = len(self.path)-1))
            
            coor = []
            for i in range(0,framecount-1):
                coor.append(path[indx[i]])
    
            self.movement = (tf,coor)
            print('Continuous path generated and saved as movement...')
       
 #############################################################################           
    #constant acceleration mode     
       if mode == 'a':  
            #important vectors       
            t =  [p[0] for p in idxlist]   #Time datapoints reported by user
            x =  [p[1] for p in idxlist]   #Index datapoints reported by user
        
            v     = np.zeros(len(x))
            a     = np.zeros(len(x))
            
            #Velocity and Acceleration at user's datapoints
            v[0] = 0   
            for i in range (0,len(x)-1):
               dx   = x[i+1] - x[i]
               dt   = t[i+1] - t[i]  
               if dt == 0: 
                   dt = .01
               a[i] = 2*(dx - v[i]*dt) / (dt*dt)
               v[i+1] = a[i]*dt + v[i]
            
                     
            videolength   = t[len(t)-1] #seconds
            fps           = 60 #fp seconds 
            framecount = (fps * videolength)
            indexmap = np.zeros(framecount,np.uint16)
            
            tf     =  np.zeros(framecount) 
            xf     =  np.zeros(framecount) 
            vf     =  np.zeros(framecount)
            af     =  np.zeros(framecount)
            xf[0] = x[0]
            #Timestamp at each frame
            for i in range (0,framecount):
                tf[i] = i/60
                
            #Saves the index that is associated with the acc and vel for a certain time segment
            for i in range (0,framecount):
                for j in range (1,len(t)):
                    if tf[i] < t[j]:
                        indexmap[i] = j - 1
                        break
                    elif tf[i] > t[j]:
                        indexmap[i] = j
            indx_max = indexmap[framecount-1]
                 
            '''
            The above code is tricky. The inner for loop searches the user's timestamps to find the correct
            index for the next code to reference. If the program's continuous time is before a timestamp, 
            the index before the timestamp is used. This causes a break and the next cont. time is used 
            to find another index. The elif condition is for any time comparisons after the final timestamp.
            This condition happens OFTEN, as many indexes are passed before reaching later times 
            in the for loop. However, the value it writes is overwritten. When the if statement finally 
            becomes true, the correct index is written, and the loop breaks. For this reason it is 
            important that it doesn't have a break statement. If it did, the if statement would have almost
            no impact on the recorded index table. 
            
            When given coordinates with constant velocity, a stop go motion is seen. Need to look at 
            data points and determine whether to use a const acceleration model or a const velocity model
            for individual segments.             
            '''

            #Velocity at each frame
            for i in range (0,framecount):
                indx = indexmap[i]
                
                if indx == indx_max:
                    s1 = 0
                    s2 = 1
                else: 
                    s1 = 1
                    s2 = 0
                    
                dt    = float(t[indx + s1]) - float(t[indx - s2])  #time between user's data points
                vdif = v[indx + s1] - v[indx] 
                if dt == 0:
                   #find next non zero dt and then calculate 
                   
                   vf[i] = vf[i-1] 
                else:    
                   vf[i] = v[indx] + (vdif * ((tf[i]) % (dt)) / (dt)) #velocity ratio of time passed
                
            
            #acceleration at each frame
            for i in range (0,framecount):
                indx = indexmap[i]
                af[i] = a[indx]
            
            #position at each frame       
            for i in range (0,framecount-1):
               indx = indexmap[i]
               dt = tf[i + 1] - tf[i]
               xf[i + 1] = 1/2*af[i] * dt*dt + vf[i]*dt + xf[i]
 #############################################################################
 
    #################################################################
    def savevideo(self,background = None,savename = 'video.avi'):            
    #################################################################   
       
       if background is None:
          background = self.cleanimg
           
       #videos can't be larger than 2GB, so videos will be saved in chunks
       points = self.movement
       time  =     points[0]   #continuous time derived from user
       coor  =     points[1]   #continuous location derived from user
       img   = cv2.imread(background) 
       height, width, layers = img.shape    
                     
       videolength   = time[len(time)-1] #seconds
       fps           = 60 #fp seconds 
       framecount    = int(fps * videolength)  
       framecap      = 3000   
       frameranges   = FrameRanges(framecount,framecap)
       n = 0
       
       for frm in frameranges:
           name = "n"+str(n)+"_"+savename
           video = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))

           for i in range (frm[0],frm[1]): 
              frame = cv2.imread(background)
              cv2.circle(frame,(coor[i][1],coor[i][0]), 15, (0,0,0), -1)
              cv2.circle(frame,(coor[i][1],coor[i][0]), 12, (255,255,255), -1)
              print('Writing frame ' +  str(i))
              video.write(frame)
               
           video.release()
           cv2.destroyAllWindows()
           n += 1
           
       print('Animation saved. Program done.')
       
        
def EstimateIndexAtTime(requestedtime, timestamps, mode = 'v', looping = False, maxindex = 100):
    timelist   = [x[0] for x in timestamps] #needs to return first column of timestamps
    tslength     = len(timelist)-1
    indexlist  = [x[1] for x in timestamps] #needs to return second column of timestamps
    maxtime    = np.amax(timelist)
    mintime    = np.amin(timelist)

    #find timestamp region to use in calculations
    if   requestedtime < mintime:
       return -1 #region = 0
   #***********
       #When time stamps don't start at zero, the dot stalls at the start. I want it
       # to stall at the first timestamp location.
   #***********    
       
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
    t0      =  timelist[region]
    tf      =  timelist[region+1]
    
    #Take the shortest path on the index loop from index_0 to index_f
    dindex = index_f - index_0
    
    if looping:
        d1 = index_f - index_0                    #index distance in one direction
        d2 = (maxindex - abs(d1)) * -np.sign(d1)  #index distance the other way around
    
        if abs(d1) < abs(d2):
            dindex = d1
        
        elif abs(d2) < abs(d1):
            dindex = d2        
                
    dt            = tf - t0
    rateofchange  = dindex / dt
           
    #estimate index for the requested time
    index = int(index_0 + rateofchange * (requestedtime - t0))
    
    #looping past min/max indices
    if looping:
       index = index % maxindex # keeps the index between 0 and max
       
    return index 

def FrameRanges(framecount,framecap):
   n_videos = int(np.ceil(framecount /framecap)) 
   framesets = []
   for n in range(0,n_videos):
           startingframe = n*framecap
           if framecount - startingframe >= framecap:
               endingframe = startingframe + framecap
           else:
               endingframe = startingframe + framecount % framecap 
           framesets.append([startingframe,endingframe])
   return framesets

      
''' 
##############################################################################
#Main
##############################################################################
'''

A = PathApp('test1.png','test1.png',[0,0,255], autogenerate = True, loop = True, mode = 'v')
#bad for debugging because A isn't defined until completed.

'''
A.ColorDistanceFilter(mode = 'BW', threshold = 220)   
A.skelatonize()          
A.CoordinateFilter() 

A.LonePointFilter(ROI = 8, n_min = 7)
A.NeighborFilter(ROI = 10, preview = False) 
A.createpath()
A.TimeStamps(mode = "typeing")

A.FormContinuousPath(mode = 'v')   
'''
'''        
A.savevideo()     #background = 'test1.png',savename = 'video1.avi'       


B1 = A.colordistance
B2 = A.skelaton
B3 = A.unorderedpath
B4 = A.startingpoints
B5 = A.path
B6 = A.timestamps

A.draw(A.colordistance)
A.draw(A.skelaton)
A.drawpoints(A.unorderedpath,1, gradient = True)
A.drawpoints(A.startingpoints,5, gradient = False)
A.draw(A.path, 3)


cv2.destroyAllWindows()
#print(A.imgname)

#A.draw(A.colordistance)
'''


#Line 718
#Get acceleration working
#create Exe. file