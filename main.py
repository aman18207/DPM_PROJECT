import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import time
import pandas as pd

from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import csv
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from moviepy.editor import *

from bokeh.plotting import figure , show,output_notebook,output_file,save
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
from bokeh.io import curdoc
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.io import export_png
from bokeh.io import curdoc
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template
pd.options.mode.chained_assignment = None
from bokeh.embed import autoload_static
from bokeh.resources import CDN





@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_video():
	# if 'file' not in request.files:
	# 	flash('No file part')
	# 	return redirect(request.url)
	file = request.files['file']
	# if file.filename == '':
	# 	flash('No image selected for uploading')
	# 	return redirect(request.url)
	# else:
	# filename = secure_filename(file.filename)
	filename = "uploaded_file.mp4"
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	#print('upload_video filename: ' + filename)
	flash('Video successfully uploaded and displayed below')
	return render_template('index.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/view',methods = ['GET','POST'])

def run():

 
	ap = argparse.ArgumentParser()

	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())



	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")


	from moviepy.editor import VideoFileClip
	clip = VideoFileClip("static/uploads/uploaded_file.mp4")
	duration = clip.duration
	# print( duration )
	# if a video path was not supplied, grab a reference to the ip camer

	# otherwise, grab a reference to the video file

	# print("[INFO] Starting the video..")
	vs = cv2.VideoCapture("static/uploads/uploaded_file.mp4")

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None


	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]
	store = []

	dict_ = {}
	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		# frame = frame[1] if args.get("input", False) else frame
		frame = frame[1] if ("static/uploads/uploaded_file.mp4", False) else frame


		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# f_check.append(frame)
		# t_check.append(t0)
		# t = np.array(frame)
		# t = t.flatten()

		# dict_[t] = t0
		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		tempvid = None
		tempvid = "static/processed/output.avi"

		if tempvid is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			writer = cv2.VideoWriter("static/processed/output.mp4", fourcc, 30,
				(W, H), True)

		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % args["skip_frames"]== 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:

				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:

					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						store.append([totalFrames,totalUp,totalDown])
						# f_temp.append(frame)
						empty.append(totalUp)
						to.counted = True


					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						store.append([totalFrames,totalUp,totalDown])
						# f_temp.append(frame)
						empty1.append(totalDown)
						#print(empty1[-1])
						
						x = []
						# compute the sum of total people inside
						x.append(len(empty1)-len(empty))
						
						# 		print("[INFO] Alert sent")

						to.counted = True


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)


		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]

                # Display the output
		for (i, (k, v)) in enumerate(info):
			# print(frame,totalUp,totalDown,x,status)
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



		# cv2.imshow("Real-Time Monitoring/Analysis Window",frame)

		# plt.imshow(frame)
		# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		# plt.show()
		writer.write(frame)
		key = cv2.waitKey(1) & 0xFF


		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		# print(totalFrames)
		fps.update()

		if config.Timer:
			# Automatic timer to stop the live stream. Set to 8 hours (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# stop the timer and display FPS information
	fps.stop()
	# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# print(totalFrames)



	filename = "log.csv"
    
	fields = ["time","exiting","entering","total people inside"]
	with open(filename, 'w') as csvfile: 
	    # creating a csv writer object 
	    csvwriter = csv.writer(csvfile) 
	        
	    # writing the fields 
	    csvwriter.writerow(fields)
	    csvwriter.writerow([0,0,0,0])
	        
	    # writing the data rows 
	    for i in range(len(store)):
	    	ttemp = store[i]
	    	ttemp[0] = (ttemp[0]/totalFrames )*duration
	    	ttemp[0]=("%.2f" % ttemp[0])
	    	ttemp.append(ttemp[2]-ttemp[1])
	    	# print(ttemp)
	    	csvwriter.writerow(ttemp)

	# # if we are not using a video file, stop the camera video stream
	# if not args.get("input", False):
	# 	vs.stop()
	#
	# # otherwise, release the video file pointer
	# else:
	writer.release()

	# close any open windows
	cv2.destroyAllWindows()
	# time.sleep(10)
	clip = VideoFileClip("static/processed/output.mp4")
	# clip.write_gif("templates/output.gif",fps = 6)
	# time.sleep(60)
	
	# return render_template('temp.html',filename = "static/processed/output.mp4")
	log = pd.read_csv("log.csv")


	time = [i for i in log["time"]]
	exiting = [i for i in log["exiting"]]
	entering = [i for i in log["entering"]]
	total = [i for i in log["total people inside"]]
	length = len(time)
	entering_coun = [0 for i in range(length)]
	exiting_coun = [0 for i in range(length)]
	# total_coun = [0 for i in range(length)]

	for i in range(1,len(time)):
		entering_coun[i] = entering[i] - entering[i-1]
		exiting_coun[i] = exiting[i] - exiting[i-1]
	max_time = max(time)
	max_time = int(max_time) + 1

	start = [i for i in range(0,max_time)]
	end = [i for i in range(1,max_time+1)]

	entering_ = [0 for i in range(max_time)]
	exiting_ = [0 for i in range(max_time)]
	total_coun = [0 for i in range(max_time)]

	for i in range(len(time)):
  		t = time[i]
  		entering_[int(t)] += entering_coun[i]
  		exiting_[int(t)] += exiting_coun[i]
  		total_coun[int(t)] = total[i]

	for i in range(1,len(total_coun)):
		if total_coun[i] == 0:
			total_coun[i] = total_coun[i-1]





	df1 = pd.DataFrame()
	df1["entering"] = entering_
	df1["left"] = start
	df1["right"] = end


	df2 = pd.DataFrame()
	df2["exiting"] = exiting_
	df2["left"] = start
	df2["right"] = end

	df3 = pd.DataFrame()
	df3["total"] = total_coun
	df3["left"] = start
	df3["right"] = end

	curdoc().theme = 'dark_minimal'
	src1 = ColumnDataSource(df1)
	p = figure(plot_height = 400, plot_width = 400, 
			title = 'Count of people entering',
			x_axis_label = 'Range', 
			y_axis_label = 'Count')

	# Add a quad glyph
	p.quad(bottom=0, top='entering', left='left', right='right', source=src1,
		fill_color='red', line_color='black', fill_alpha = 0.75,
		hover_fill_alpha = 1.0, hover_fill_color = 'navy')


	h = HoverTool(tooltips = [('Count', '@entering'),
							('Interval', '[@left,@right]')])

	p.add_tools(h)
# output_notebook()
# show(p)
	src2 = ColumnDataSource(df2)
	p2 = figure(plot_height = 400, plot_width = 400, 
			title = 'Count of people exiting',
			x_axis_label = 'Range', 
			y_axis_label = 'Count')

	# Add a quad glyph
	p2.quad(bottom=0, top='exiting', left='left', right='right', source=src2,
		fill_color='red', line_color='black', fill_alpha = 0.75,
		hover_fill_alpha = 1.0, hover_fill_color = 'navy')


	h = HoverTool(tooltips = [('Count', '@exiting'),
							('Interval', '[@left,@right]')])

	p2.add_tools(h)

	src3 = ColumnDataSource(df3)
	p3 = figure(plot_height = 400, plot_width = 400, 
			title = 'Count of people inside',
			x_axis_label = 'Range', 
			y_axis_label = 'Count')

	# Add a quad glyph
	p3.quad(bottom=0, top='total', left='left', right='right', source=src3,
		fill_color='red', line_color='black', fill_alpha = 0.75,
		hover_fill_alpha = 1.0, hover_fill_color = 'navy')


	h = HoverTool(tooltips = [('Count', '@total'),
							('Interval', '[@left,@right]')])

	p3.add_tools(h)
	from bokeh.layouts import column,row
	temp = (row(p,p2,p3))
	script1,div1 = components(temp)
	cdn_js = CDN.js_files
	cdn_css = CDN.css_files
	return render_template('homepage.html',script1 = script1,div1 = div1,cdn_css = cdn_css,cdn_js  = cdn_js)




@app.route('/video',methods = ['GET','POST'])
def show_video():
	
	return render_template('video.html')	

if __name__ == "__main__":
    app.run(debug = True)