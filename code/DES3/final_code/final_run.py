import os

os.system("python3 capture_for_testrun.py")

os.system("python3 live_depth_box2.py")

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

from read_det_result import *
locations = read_file()
print("################read_completed###############")
startPointX = locations[0,0]
startPointY = locations[0,1]

tissue_endX = locations[1,0]
tissue_endY = locations[1,1]
tissue_height = locations[1,2]

bottle_endX = locations[2,0]
bottle_endY = locations[2,1]
bottle_height = locations[2,2]

obstacle_endX = locations[3,0]
obstacle_endY = locations[3,1]

print("################Start_planning_path###############"+str(startPointX))

from path_planning_module import *

scaleX = 295
scaleY = 302

#############default###############
#obstaclesMap = np.array([#[2.0, 0.3][1.4, 0.6]])
#startPointX = 895
#startPointY = 413
#targetPointX = 225
#targetPointY =480
###################################

obstaclesMap = np.array([
                         [obstacle_endX/scaleX, (540-obstacle_endY)/scaleY]
               ])

if target == "bottle":
    targetPointX = bottle_endX
    targetPointY = bottle_endY
    endx = (targetPointX)/scaleX
    endy = (540-targetPointY)/scaleY

if target == "tissue":
    targetPointX = tissue_endX
    targetPointY = tissue_endY
    endx = (targetPointX)/scaleX
    endy = (540-targetPointY)/scaleY

startx = startPointX/scaleX
starty = (540-startPointY)/scaleY
endx = (targetPointX)/scaleX
endy = (540-targetPointY)/scaleY


carPath = genarate_path(startx,starty,135.0,endx,endy, True,obstaclesMap)

print("################planning_path_compelete###############"+str(len(carPath)))

from car_control_module import *
control_by_map(carPath,1)


from playsound import playsound

height = 0
if target == "bottle":
    height = bottle_height
if target == "tissue":
    height = tissue_height

positionAt = 0 # 0:at waist, 1: above waist, 2: at floor, 5:error
if height == 0:
    positionAt = 5
elif height < 60:
    positionAt = 2
elif height < 120:
    positionAt = 0
elif height < 150:
    positionAt = 1

if positionAt == 0:
    playsound('./at.mp3')
elif positionAt == 1:
    playsound('./above.mp3')
elif positionAt == 2:
    playsound('./floor.mp3')

from car_control_module import *
delaySecond(5)
control_by_map(carPath,0)
