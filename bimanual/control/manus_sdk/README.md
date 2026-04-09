# Manus Linux SDK

## Required hardware
1. Windows 10 machine for connecting with MANUS gloves
2. Linux machine for connecting with robot
3. 2x Manus gloves
4. Quest3 headset with 2x controllers

## Network Setup
1. Connect Windows machine and Linux machine to switch via ethernet. 
2. Configure a static IP on both machines for the switch network interface.
3. Now you should be able to ping from one machine to another. You may need to disable Windows network protective features. 

## Sending glove data from Windows to Linux
### Windows 10 machine
1. Install [MANUS Core SDK](https://www.manus-meta.com/resources/downloads/quantum-metagloves) for Windows 10.
2. Connect MANUS wireless dongle via USB.
3. Turn on both gloves. The flashing LED on gloves should go from blinking to solid once connected. 
4. Connect gloves through Manus Core UI. Calibrate gloves if needed.
### Linux machine
5. On Linux machine, build docker container.
```
$ docker build -t manus .
```
6. Launch docker container in interactive mode with this directory as a volume.
```
$ docker run --network=host -v .:/home -it manus /bin/bash
```
7. Build Manus C++ client
```
$ make all
```
8. In a tmux session start the client running in the background
```
$ ./SDKMinimalClient.out
```
9. The client will publish ZeroMQ pub/sub JSON messages of glove data to `tcp://localhost:5555`. See `control/manus.py` for code that reads glove data from port 5555. 
