import os

target = "none"

while target != "tissue" and target != "bottle":

	os.system("python3 voice_recording_module.py")
	os.system("python3 voice_convert_module.py")
	os.system("python3 voice_recognition_module.py")

	file1 = open("voice_result.txt","r")
	target = file1.read()

	print("################################")
	print(target)
	print("##################################")
