import time
import serial
import numpy as np

ser = 0
    
def control_by_map(tr,forward,forwardSpeed = 69.0, turnSpeed = -20.0):

    
    def turn(angleInDegree):
        angle = angleInDegree * 10
        if angle > 400:
            angle = 400
        if angle < -400:
            angle = -400
        i = 1500 + angle
        #print(i)
        ser.write((str(i)+"\r\n").encode())
        delay(0.01)
        
    def run(velPercentage):
        vel = velPercentage * 4
        if vel > 400:
            vel = 400
        if vel < -400:
            vel = -400
        i = 4500 + vel
        ser.write((str(i)+"\r\n").encode())
        
    def run(velPercentage):
        vel = velPercentage * 4
        if vel > 400:
            vel = 400
        if vel < -400:
            vel = -400
        i = 4500 + vel
        ser.write((str(i)+"\r\n").encode())
        
    def delay(timeInSecond):
        t = time.time()
        counter = 0
        while time.time() < t + timeInSecond:
            counter = counter + 1
        
    def stop():
        delay(0.01)
        ser.write("5000\r\n".encode())
        delay(0.01)
        ser.write("5000\r\n".encode())
        delay(0.01) 
        
    def beep(mode):#0 = off,1 = on, 2 = slow blink, 3= fast blink
        if mode == 0:
            ser.write("6000\r\n".encode())
        elif mode == 1:
            ser.write("6001\r\n".encode())
        elif mode == 2:
            ser.write("6002\r\n".encode())
        elif mode == 3:
            ser.write("6003\r\n".encode())  
        
    length, di = np.shape(tr)
    
    print("Start control the Car with total ", length, " steps!")

    ser=serial.Serial("/dev/ttyUSB0", 38400)
    #ser = serial.Serial('COM5', 38400)
    
    beep(3)
    
    if forward == 0:
        forwardSpeed = -forwardSpeed
        for x in range(length):
            turn(tr[length-2-(x-1),4]*turnSpeed)
            run(tr[length-2-(x-1),3]*forwardSpeed)
            delay(0.3)
            print("step",x+1)
    else:
    
        for x in range(length):
            turn(tr[x-1,4]*turnSpeed)
            run(tr[x-1,3]*forwardSpeed)
            delay(0.3)
            print("step",x+1)

    beep(1)
    turn(0)
    run(0)
    delay(3)
    beep(0)
    delay(0.1)
    stop()
    ser.close()
    

def delaySecond(timeInSecond):
    t = time.time()
    counter = 0
    while time.time() < t + timeInSecond:
        counter = counter + 1


    

        

    

    
