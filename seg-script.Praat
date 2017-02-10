# This script extracts the mean and max intensity of an audio file
# and the mean and max value of each of the first two formants.
# It then writes these values out to a file

# Give Praat the path to the directory where your audio files are stored
# ASSIGN THIS VARIABLE TO THE PATH TO YOUR AUDIO DATA DIRECTORY
directory$ = "./Dialects/wav/"

# Give Praat the path to your output file
# ASSIGN THIS VARIABLE TO THE FULL PATH TO YOUR OUTPUTFILE
outputfile$ = "./wrd-feats.csv"
deleteFile: outputfile$ 

# print a line to your output file containing the names of the features
appendFileLine: outputfile$, "maxint,minint,meanint,maxf1,minf1,meanf1,maxf2,minf2,meanf2,maxf0,minf0,meanf0,jitter,shimmer,maxhnr,minhnr,meanhnr,dx"

# loop through the files one by one
strings = Create Strings as file list: "list", directory$ + "/*.wav"
numberOfFiles = Get number of strings
for ifile to numberOfFiles
    selectObject: strings
    fileName$ = Get string: ifile
    appendInfoLine: fileName$

    # this is just so you know the name of the sound you're dealing with
    # that way, you can select it later on if you need to
    fileIntro$ = replace$ (fileName$, ".wav", "", 1)
    appendInfoLine: fileIntro$

    # read in the sound file
    Read from file: directory$ + "/" + fileName$
	Read from file: directory$ + "/" + fileIntro$ + ".TextGrid"
	fileIntro$ = replace$ (fileIntro$, ".", "_", 1)
    appendInfoLine: fileIntro$ + " read!"

    
    selectObject: "Sound " + fileIntro$
    To Intensity: 100, 0, "yes"

    selectObject: "Sound " + fileIntro$
    To Formant (burg): 0, 5, 5500, 0.025, 50
    
    selectObject: "Sound " + fileIntro$
	To Pitch (ac): 0, 75, 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, 600
    To PointProcess
	
	selectObject: "Sound " + fileIntro$
	To Harmonicity (ac): 0.01, 75, 0.1, 4.5
	
    selectObject: "TextGrid " + fileIntro$
    tier = 2
    nT = Get number of intervals: tier
    appendInfoLine: "Num of intervals: ", nT

    for p to nT
        selectObject: "TextGrid " + fileIntro$

        label$ = Get label of interval: tier, p
        if label$ != "sp"
            pStart = Get starting point... tier p
            pEnd = Get end point... tier p

            selectObject: "Intensity " + fileIntro$
            maxintensity = Get maximum: pStart, pEnd, "Parabolic"
            minintensity = Get minimum: pStart, pEnd, "Parabolic"
            meanintensity = Get mean: pStart, pEnd, "energy"
            appendFile: outputfile$, maxintensity, ",", minintensity, ",", meanintensity, ","

            selectObject: "Formant " + fileIntro$
            for formnum from 1 to 2
                maxf = Get maximum: formnum, pStart, pEnd, "Hertz", "Parabolic"
                minf = Get minimum: formnum, pStart, pEnd, "Hertz", "Parabolic"
                meanf = Get mean: formnum, pStart, pEnd, "Hertz"
                appendFile: outputfile$, maxf, ",", minf, ",", meanf, ","
            endfor

            selectObject: "Pitch " + fileIntro$
            maxf0 = Get maximum: pStart, pEnd, "Hertz", "Parabolic"
            minf0 = Get minimum: pStart, pEnd, "Hertz", "Parabolic"
            meanf0 = Get mean: pStart, pEnd, "Hertz"
            appendFile: outputfile$, maxf0, ",", minf0, ",", meanf0, ","

            selectObject: "PointProcess " + fileIntro$
            jitter = Get jitter (local): pStart, pEnd, 0.0001, 0.02, 1.3
            selectObject: "Sound " + fileIntro$
            plusObject: "PointProcess " + fileIntro$
            shimmer = Get shimmer (local): pStart, pEnd, 0.0001, 0.02, 1.3, 1.6
            appendFile: outputfile$, jitter, ",", shimmer, ","

            selectObject: "Harmonicity " + fileIntro$
            maxhnr = Get maximum: pStart, pEnd, "Parabolic"
            minhnr = Get minimum: pStart, pEnd, "Parabolic"
            meanhnr = Get mean: pStart, pEnd
            appendFile: outputfile$, maxhnr, ",", minhnr, ",", meanhnr, ","

            if startsWith(fileIntro$, "southern")
        	     appendFileLine: outputfile$, "1"        
            else
        	     appendFileLine: outputfile$, "0"        
            endif
        endif
    endfor
	
  	removeObject: "Sound " + fileIntro$
	removeObject: "TextGrid " + fileIntro$
	removeObject: "PointProcess " + fileIntro$
	removeObject: "Formant " + fileIntro$
	removeObject: "Intensity " + fileIntro$
    removeObject: "Pitch " + fileIntro$
    removeObject: "Harmonicity " + fileIntro$
endfor


