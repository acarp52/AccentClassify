# This script extracts the mean and max intensity of an audio file
# and the mean and max value of each of the first two formants.
# It then writes these values out to a file

# Give Praat the path to the directory where your audio files are stored
# ASSIGN THIS VARIABLE TO THE PATH TO YOUR AUDIO DATA DIRECTORY
directory$ = "/Users/sushantkafle/Dropbox/Courses/Spring17/SLT/Hackathon1/Dialects"

# Give Praat the path to your output file
# ASSIGN THIS VARIABLE TO THE FULL PATH TO YOUR OUTPUTFILE
outputfile$ = "/Users/sushantkafle/Dropbox/Courses/Spring17/SLT/Hackathon1/dialects_.csv"
deleteFile: outputfile$ 

# print a line to your output file containing the names of the features
appendFileLine: outputfile$, "maxint,meanint,maxf1,meanf1,maxf2,meanf2,maxf0,jitter,shimmer,hnr,dx"

# loop through the files one by one
strings = Create Strings as file list: "list", directory$ + "/*.mp3"
numberOfFiles = Get number of strings
for ifile to numberOfFiles
    selectObject: strings
    fileName$ = Get string: ifile
    appendInfoLine: fileName$

    # this is just so you know the name of the sound you're dealing with
    # that way, you can select it later on if you need to
    fileIntro$ = replace$ (fileName$, ".mp3", "", 1)
    appendInfoLine: fileIntro$

    # read in the sound file
    Read from file: directory$ + "/" + fileName$
    appendInfoLine: fileIntro$ + "read!"

    # make sure you've selected the sound
    selectObject: "Sound " + fileIntro$

    # get max and mean intensity (a.k.a. energy) of that sound
    To Intensity: 100, 0, "yes"
    appendInfoLine: fileIntro$ + "intensity!"

    maxintensity = Get maximum: 0, 0, "Parabolic"
    meanintensity = Get mean: 0, 0, "energy"

    # print the info out to a file
    appendFile: outputfile$, maxintensity, ",", meanintensity, ","

    # make sure you select the sound again
    selectObject: "Sound " + fileIntro$

    # get formant information for that sound
    To Formant (burg): 0, 5, 5500, 0.025, 50
    appendInfoLine: fileIntro$ + " formant!"

    for formnum from 1 to 2
			 appendInfoLine: "formant"
         maxf = Get maximum: formnum, 0, 0, "Hertz", "Parabolic"
         meanf = Get mean: formnum, 0, 0, "Hertz"
	 		 appendFile: outputfile$, maxf, ",", meanf, ","
    endfor
	
	# get F0
	selectObject: "Sound " + fileIntro$
	To Pitch (ac): 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600
	maxf0 = Get maximum: 0, 0, "Hertz", "Parabolic"
	
	
	# get jitter, shimmer
	To PointProcess
	jitter = Get jitter (local): 0, 0, 0.0001, 0.02, 1.3
	selectObject: "Sound " + fileIntro$
	plusObject: "PointProcess " + fileIntro$
	shimmer = Get shimmer (local): 0, 0, 0.0001, 0.02, 1.3, 1.6

	selectObject: "Sound " + fileIntro$
	To Harmonicity (ac): 0.01, 75, 0.1, 4.5
	hnr = Get mean: 0, 0
	appendFile: outputfile$, maxf0, ",", jitter, ",", shimmer, ",", hnr, ","

    appendInfoLine: "data complete!"

    # finally, write out the class (northern or southern)
	  ## IF YOU WANT TO USE WEKA COMMENT THIS OUT
    if startsWith(fileIntro$, "southern")
	     appendFileLine: outputfile$, "1"        
    else
	     appendFileLine: outputfile$, "0"        
    endif

	  ## IF YOU WANT TO USE WEKA UNCOMMENT
    #if startsWith(fileIntro$, "southern")
	  #   appendFileLine: outputfile$, "south"        
    #else
	  #    appendFileLine: outputfile$, "north"        
    #endif


  	 removeObject: "Sound " + fileIntro$
	 removeObject: "Formant " + fileIntro$
	 removeObject: "Intensity " + fileIntro$


endfor


