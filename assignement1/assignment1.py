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



# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper
    #cap is the videocapture object of the input video
    #lower and upper are the time boundaries of the video, in  [ms] 
    

def main(input_video_file: str, output_video_file: str) -> None:
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
    #we define the below frame width and height
    #they are smaller than the input_frame_width and height 
    #the idea is to downsample to keep the file size small
    output_frame_width = 540
    output_frame_height = 380
    #writing the output video file
    #if we want to write only gray scale part of the video 
    #the argument isColor = False specify that the output is in Gray scale and not RGB
    out = cv2.VideoWriter(output_video_file, fourcc, input_fps, (output_frame_width, output_frame_height))
    print("fps is", input_fps)
    print("frame size is :", input_frame_width, input_frame_height)
    # while loop where the real work happens
    while cap.isOpened():
        #capture frame by frame
        ret, frame = cap.read()
        #before working on the frame, we resize it to the same size as the output
        frame = cv2.resize(frame, (output_frame_width, output_frame_height), fx = 0, fy = 0,
                         interpolation = cv2.INTER_CUBIC)
        if ret:
            # font,colour, ... to be used for video text annotation
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = (output_frame_width * output_frame_height) / (1000 * 1000)
            color = (0, 255, 255)
            thickness = 2
            # define q as the exit button
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 0, 2000):
                #convert frame to gray scale
                output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #to be able to write the video, we have to convert from gray to RGB
                #it will stay in gray, but the function videoWrite needs to have an RGB input
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
                # Use cv2.putText(frame, Text, org, font, color, thickness) method for inserting text on video
                cv2.putText(output, 'Gray scale', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap, 2000, 4000):  
                #frame does not change
                output = frame
                cv2.putText(output, 'RGB scale', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,4000,6000):
                #apply GaussianBlur(input, kernel_size, sigma_x, sigma_y)
                output = cv2.GaussianBlur(frame, (5,5), 1, 1, cv2.BORDER_DEFAULT)
                cv2.putText(output, 'Gaussian, kernel = (5,5)', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,6000,8000):
                #apply GaussianBlur(input, kernel_size, sigma_x, sigma_y)
                output = cv2.GaussianBlur(frame, (13,13), 1, 1, cv2.BORDER_DEFAULT)
                cv2.putText(output, 'Gaussian, kernel = (13,13) ; image is more smoothed', (50, 50),font, 1
                , color, thickness, cv2.LINE_4)

            if between(cap,8000,10000):
                #apply cbilateralFilter(input,output,kernel_size,sigmaColor,sigmaSpace,borderType=BORDER_DEFAULT)
                output = cv2.bilateralFilter(frame,9,10,10)
                cv2.putText(output, 'bilateral filtering, sigma=10', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            if between(cap,10000,12000):
                #apply cbilateralFilter(input,output,kernel_size,sigmaColor,sigmaSpace,borderType=BORDER_DEFAULT)
                output = cv2.bilateralFilter(frame,9,100,100)
                cv2.putText(output, 'bilateral filtering,sigma=100', (50, 50),font, 1, color, thickness, cv2.LINE_4)
            # write frame that you processed to output
            out.write(output)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)
            cv2.imshow('Output', output)

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

main("input.mp4", "output.mp4")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='OpenCV video processing')
#     parser.add_argument('-i', "--input", help='full path to input video that will be processed')
#     parser.add_argument('-o', "--output", help='full path for saving processed video output')
#     args = parser.parse_args()

#     if args.input is None or args.output is None:
#         sys.exit("Please provide path to input and output video files! See --help")

