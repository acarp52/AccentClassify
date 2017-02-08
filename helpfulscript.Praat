# This script extracts the mean and max intensity of an audio file
# and the mean and max value of each of the first four formants.
# It then writes these values out to a file

# Give Praat the path to the directory where your audio files are stored
# ASSIGN THIS VARIABLE TO THE PATH TO YOUR AUDIO DATA DIRECTORY
directory$ = "C:\Users\acarp\Documents\Speech Processing\Hackathon1\Dialects"

# Give Praat the path to your output file
# ASSIGN THIS VARIABLE TO THE FULL PATH TO YOUR OUTPUTFILE
outputfile$ = "C:\Users\acarp\Documents\Speech Processing\Hackathon1\dialects.csv"
deleteFile: outputfile$ 

# print a line to your output file containing the names of the features
appendFileLine: outputfile$, "maxint,meanint,maxf1,meanf1,maxf2,meanf2,dx"

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

    appendInfoLine: "data complete!"

    # finally, write out the class (northern or southern)
    if startsWith(fileIntro$, "southern")
	     appendFileLine: outputfile$, "1"        
    else
	     appendFileLine: outputfile$, "0"        
    endif

  	 removeObject: "Sound " + fileIntro$
	 removeObject: "Formant " + fileIntro$
	 removeObject: "Intensity " + fileIntro$


endfor


