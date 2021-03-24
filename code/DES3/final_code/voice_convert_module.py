from pydub import AudioSegment

file = AudioSegment.from_wav('before.wav')
total = len(file)
file = file[total*0.95-1100:total*0.95]
file.export('after.wav')
