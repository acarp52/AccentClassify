selectObject: "Sound southern-english44_mp3"
To Intensity: 100, 0, "yes"

selectObject: "TextGrid southern-english44_mp3"
nT1 = Get number of intervals: 2
for p to nT1
selectObject: "TextGrid southern-english44_mp3"
pStart = Get starting point... 2 p
pEnd = Get end point... 2 p
selectObject: "Intensity southern-english44_mp3"
maxint = Get maximum: pStart, pEnd, "Parabolic"
appendInfoLine: maxint
endfor

