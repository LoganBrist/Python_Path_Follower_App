# Python_Path_Follower_App

The Python program calculates continuous movement along a user-drawn path on an image based on user-provided time and position, and animates it using OpenCV. 
The path pixels are first found with color filtering, and then they are ordered, skelatonizes, and filtered so the animated object doesn't jump from point to point.

This was done to pair a graphic of a tram's movement with video of the tram timelapse in the Dallas/Fort Worth International Airport Skytram, however it can easily be extended 
into different dimensions. The object's position is parameterized by the path's index, so take a path drawn on a 1D line,
a 2D surface like this map, or a 3D environment, and you would still have the core operation of cycling through an index at varying velocities or
accelerations in the same way seen here.



