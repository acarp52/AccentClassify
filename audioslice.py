from pydub import AudioSegment

sound = AudioSegment.from_mp3("./Dialects/northern-english49.mp3")

# len() and slicing are in milliseconds
halfway_point = len(sound) / 2
second_half = sound[halfway_point:]

# Concatenation is just adding
second_half_3_times = second_half + second_half + second_half

# writing mp3 files is a one liner
second_half_3_times.export("/path/to/new/file.mp3", format="mp3")