import os
from pydub import AudioSegment


def main():

    for filename in os.listdir("Dialects/mp3"):
        sound = AudioSegment.from_mp3("Dialects/mp3/{0}".format(filename))

        # len() and slicing are in milliseconds
        halfway_point = len(sound) / 2
        timeSlice = len(sound) / 20
        slices = 1
        startTime = 0
        endTime = timeSlice

        while slices <= 20:
            newSlice = sound[startTime:endTime]
            newSlice.export("Dialects/mp3/{0}_{1}.mp3".format(filename[:-4], slices), format="mp3")
            startTime = endTime
            endTime = endTime + timeSlice
            slices += 1

main()