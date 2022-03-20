"""
Created on Sun Mar 13 2022

@author: deniz
"""

#skeleton cloned from https://github.com/gourie/opencv_video/blob/main/opencv_process_video.py
#assignment for Computer Vision course at KU Leuven

#while running the code, you can press "q" to exit it 

import argparse
import cv2
import sys
import numpy as np



"""
some definitions of variables :
"""
#we define the below frame width and height of the frame frame
#they are smaller than the input_frame_width and height 
#the idea is to downsample to keep the file size small
output_frame_width = 540
output_frame_height = 380
# font,colour, ... to be used for video text annotation
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
font_size = (output_frame_width * output_frame_height) / (1000 * 1000)
color = (0, 255, 255)
thickness = 1

"""
functions to use later
"""
# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
    #cap is the videocapture object of the input video
    #lower and upper are the time boundaries of the video, in  [ms] 

#function to open the video as a video capture object and write the output video
def OpenVideoAndWriteOutput(input_video_file,output_video_file):
    # OpenCV video objects to work with
    #create a videocapture object from the input video.
    cap = cv2.VideoCapture(input_video_file)      
    #here, we get the fps of the input video
    input_fps = int(round(cap.get(5)))          
    #here, we get the width of the input video
    input_frame_width = int(cap.get(3))
    #here, we get the height of the input video                   
    input_frame_height = int(cap.get(4))
    # saving output video as .mp4                  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    #writing the output video file
    #if we want to write only gray scale part of the video 
    #the argument isColor = False specify that the output is in Gray scale and not RGB
    
    #if you do not want to have the output dimensions as the same as input dimensions, comment the 2 lines below
    # output_frame_width = input_frame_width
    # output_frame_height = input_frame_height
    
    out = cv2.VideoWriter(output_video_file, fourcc, input_fps, (output_frame_width, output_frame_height))
    print("fps is", input_fps)
    print("frame size is :", input_frame_width, input_frame_height)
    return cap,out
    

"""
###################################################################
######## Definition of Functions : Basic Image Processing #########
###################################################################

"""


def toGray(input):
    """
    convert frame to gray 
    """
    #convert frame to gray scale
    frame = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    #to be able to write the video, we have to convert from gray to RGB
    #it will stay in gray, but the function videoWrite needs to have an RGB input
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

def noChange(input):
    """
    video stays in RGB
    """
    frame = input
    return frame

def gaussianBlur(input, kernel = 5):
    """
    apply gaussian blur
    """
    #apply GaussianBlur(input, kernel_size, sigma_x, sigma_y)
    frame = cv2.GaussianBlur(input, (kernel,kernel), 1, 1, cv2.BORDER_DEFAULT)
    return frame


def bilateralFiltering(input,kernel=9,sigma=10):
    """
    apply bilateral Filtering
    """
    #apply cbilateralFilter(input,output,kernel_size,sigmaColor,sigmaSpace,borderType=BORDER_DEFAULT)
    frame = cv2.bilateralFilter(input,kernel,sigma,sigma)
    return frame

def thresholdingColorHSV(input,lower_threshold = np.array([50,150,0]), upper_threshold = np.array([100,255,255])):
    """
    threshold on color, in HSV space
    
    np.array([110,50,50]) and np.array([130,255,255] is for BLUE
    """
    # conversion of BGR to HSV
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    # Here we are defining range of color in HSV
    frame = cv2.inRange(hsv, lower_threshold, upper_threshold)
    #convert to BGR to save as video
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

def thresholdingColorHSV_Dilation(input,lower_threshold = np.array([50,150,0]), upper_threshold = np.array([100,255,255])):
    """
    threshold on color, in HSV space with dilation
    we apply dilation to have a better output
    see :  https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/#:~:text=Morphological%20operations%20are%20simple%20transformations,as%20well%20as%20decrease%20them.
    OR : https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        """
    # conversion of BGR to HSV
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    # Here we are defining range of bluecolor in HSV
    frame = cv2.inRange(hsv, lower_threshold, upper_threshold)
    #do dilation
    frame = cv2.dilate(frame.copy(), (8,8), iterations=20)
    #convert to BGR to save as video
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


"""
###################################################################
######## Definition of Functions : Object Detection ###############
###################################################################

"""


def sobelDetector(input, ksize =5, axis='x'):
    # conversion of BGR to GRAY
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    #smooth to reduce noise
    smoothed = gaussianBlur(gray,kernel=45)
    #sobel edge detection on the x axis 
    sobel_x = cv2.Sobel(src=smoothed, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    #sobel edge detection on the y axis 
    sobel_y = cv2.Sobel(src=smoothed, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    #sobel edge detection on the x AND y axis
    sobel_xy = cv2.Sobel(src=smoothed, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)
    
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)

    grad = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    
    
    if axis =='x':
        frame = sobel_x
    elif axis == 'y':
        frame = sobel_y
    elif axis == 'xy':
        frame = grad
    else:
        frame = input     
    
    #sobel output is in range 0...1 byt videoWrite wants value in range 0...255
    frame = (255*frame).clip(0,255).astype(np.uint8)
    #convert to BGR to save as video
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    return frame
    
def houghTransformCircles(input,param1,param2,minRadius,maxRadius):
    frame = input
    # conversion of BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #smooth to reduce noise
    smoothed = gaussianBlur(gray)
    rows = smoothed.shape[0]
    #param 1 : sensitivity (strenght of edges)
    #param 2 :how many edge needed to define a circcle
    circles = cv2.HoughCircles(smoothed, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)

    return frame
    
def trackObjectByColor(input,lower_threshold = np.array([5, 75, 25]), upper_threshold = np.array([25, 255, 255])):
    """
    track an object based on its color
    based on https://pyimagesearch.com/2015/09/21/opencv-track-object-movement/
    and https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
    """
    frame = input
    # conversion of BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #create color mask 
    mask = cv2.inRange(hsv,lower_threshold,upper_threshold)
    #detect contours of the object 
    _,contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #take the largest value (if there exist other smaller object with orange value, they are ignored)
    max_contour = contours[0]
    for contour in contours:
            if cv2.contourArea(contour)>cv2.contourArea(max_contour):
                  max_contour=contour
    contour=max_contour
    approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour,True),True)
    #draw bounding rectangle
    x,y,w,h=cv2.boundingRect(approx)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
    
    return frame



def templateMatch(input,temp="template/gohan.png"):
    frame = input
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    #read template from (png file)
    template = cv2.imread(temp,0)
    
    #resize the template (conserve the widht height ratio)
    resizing_factor = 0.20    #resizing factor
    width = int(template.shape[::-1][0] * resizing_factor)
    height = int(template.shape[::-1][1] * resizing_factor)
    dim = (width, height)
    template = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)
    
    #width and height of the resized template :
    w,h = template.shape[::-1]
    
    #perform matching with SQDIFF_NORMED method
    res = cv2.matchTemplate(gray,template,cv2.TM_SQDIFF_NORMED)
    
    #get min/max values and location of the squared difference between template and image
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    #we are intereseted in the location where the template match the image : minimum squared difference --> min_loc
    top_left = min_loc
    #from the min_loc, we add the size of the template in order to have the other corner of the rectangle
    bottom_right = (top_left[0] + w, top_left[1] + h)
    #we draw the rectange from top_left to bottom_right
    cv2.rectangle(frame,top_left, bottom_right, 255, 2)

    


    return frame


"""
###################################################################################
######## Applied the previously defined functions at different time steps #########
###################################################################################
"""
    
def main(input_video_file: str, output_video_file: str) -> None:
    cap,out = OpenVideoAndWriteOutput(input_video_file,output_video_file)
    # while loop where the real work happens
    while cap.isOpened():
        #capture frame by frame
        ret, frame = cap.read()
        #uncomment the following lines if you have downsampled the image in the beginning : 
        #before working on the frame, we resize it to the same size as the output
        frame = cv2.resize(frame, (output_frame_width, output_frame_height), fx = 0, fy = 0,
                          interpolation = cv2.INTER_CUBIC)
        if ret:

            # define q as the exit button
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 0, 1000):
                frame = toGray(frame)
                # Use cv2.putText(frame, Text, org, font, color, thickness) method for inserting text on video
                cv2.putText(frame, 'Gray scale', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 1000, 2000):
                frame = noChange(frame)
                cv2.putText(frame, 'RGB scale', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 2000, 3000):
                frame = toGray(frame)
                cv2.putText(frame, 'Gray scale', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 3000, 4000):
                frame = noChange(frame)
                cv2.putText(frame, 'RGB scale', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,4000,6000):
                frame = gaussianBlur(frame)
                cv2.putText(frame, 'Gaussian, kernel = (5,5)', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,6000,8000):
                frame = gaussianBlur(frame,kernel=13)
                cv2.putText(frame, 'Gaussian, kernel = (13,13)', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'image is more smoothed', (50,70),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,8000,11000):
                frame = bilateralFiltering(frame)
                cv2.putText(frame, 'bilateral filtering, sigma=10', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'edges are preserved, thanks to', (50,70),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'the filter as a function of', (50,90),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'pixel difference', (50,110),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,11000,13000):
                frame = bilateralFiltering(frame,9,1000000)
                cv2.putText(frame, 'bilateral filtering,sigma=1000000', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'becomes similar to Gaussian', (50,70),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 13000, 15000):
                frame = noChange(frame)
            if between(cap,15000,18000):
                frame = thresholdingColorHSV(frame)
                cv2.putText(frame, 'blue thresholding in HSV space', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,18000,21000):
                frame = thresholdingColorHSV_Dilation(frame)
                cv2.putText(frame, 'blue thresholding in HSV', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'with dilation', (50,70),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 21000, 23000):
                frame = noChange(frame)
            if between(cap,23000,25000):
                frame = sobelDetector(frame, ksize =15, axis='x')
                cv2.putText(frame, 'Sobel - Vertical edges', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,25000,27000):
                frame = sobelDetector(frame, ksize =15, axis='y')
                cv2.putText(frame, 'Sobel - Horizontal edges', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,27000,29000):
                frame = sobelDetector(frame, ksize =3, axis='x')
                cv2.putText(frame, 'Sobel - Horizontal edges', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'Smaller Kernel', (50,70),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 29000, 31000):
                frame = noChange(frame)
            if between(cap,31000,36000):
                #houghTransformCircles(input,param1,param2,minRadius,maxRadius)
                houghTransformCircles(frame,180,30,1,0)
                cv2.putText(frame, 'Hough Transform, param1=180', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'param2 = 30', (50,70),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'minRadius = 1, maxRadius = 0', (50,90),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,36000,40000):
                #houghTransformCircles(input,param1,param2,minRadius,maxRadius)
                houghTransformCircles(frame,180,20,1,0)
                cv2.putText(frame, 'Hough Transform, param1=180', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'param2 = 20', (50,70),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'minRadius = 1, maxRadius = 0', (50,90),font, 1, color, thickness, cv2.LINE_4)  
            if between(cap,40000,42000):
                #houghTransformCircles(input,param1,param2,minRadius,maxRadius)
                houghTransformCircles(frame,100,15,1,10)
                cv2.putText(frame, 'Hough Transform, param1=100', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'param2 = 10', (50,70),font, 1, color, thickness, cv2.LINE_4)
                cv2.putText(frame, 'minRadius = 1, maxRadius = 15', (50,90),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,42000,47000):
                templateMatch(frame)
                cv2.putText(frame, 'Match template', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,47000,60000):
                trackObjectByColor(frame)
                cv2.putText(frame, 'Track orange Object', (50, 50),font, 1, color, thickness, cv2.LINE_4)
                
                
            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()

main("videos/input.mp4", "output.mp4")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='OpenCV video processing')
#     parser.add_argument('-i', "--input", help='full path to input video that will be processed')
#     parser.add_argument('-o', "--output", help='full path for saving processed video output')
#     args = parser.parse_args()

#     if args.input is None or args.output is None:
#         sys.exit("Please provide path to input and output video files! See --help")

