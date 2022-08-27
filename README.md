# CaptureQueen
Over-the-board chess real-time capture system.

CaptureQueen was designed on a Raspberry Pi Model 4 with the vanilla
Rasperberry Pi Camera module and the Raspberry Pi official 7" touchscreen.
A ESP32 based chess clock is also included to allow the game to have an
authentic over-the-board feel.  The components communicate through a local
MQTT server.  MQTT is a light weight messaging protocol used in many 
'internet of things' things.  The basic idea comes from karayaman's 
"Play-online-chess-with-real-chess-board", but  Unlike karayaman's 
program, CaptureQueen is meant for over-the-board chess between two humans.


#Components:
-- Networked ChessClock, sends current turn and time remainting for each 
   player.
-- Camera, images of each move are triggered by the ChessClock.  The 
   camera is mounted on a light-weight tripod for a top-view of the baord.
-- Raspberry Pi model 4 b, handles the image processing and game tracking

#Installation:
From the latest raspian, open a command line and clone this repo from github

## install mosquitto
$ sudo apt update
$ sudo apt install -y mosquitto
$ TODO: test mosquitto

## install python code onto Raspberry Pi
$ git clone https://github.com/wyolum/CaptureQueen
$ pip3 install chess
$ ...
$ TODO: pip install dependencies

Make sure the Pi is connected to the WAN.  This will be used for MQTT transport.

## TODO: assemble ChessClock
Upload the code to the ESP clock using arduino (update mqtt ip address)


$cd CaptureQueen
$python3 fen_capture.py --calibrate True
(center board in displayed image and hit the 'q' key to calibrate the board)
<IMG>
Hit the 'q' key again to quit the program


# Play Chess
## On the Raspberry Pi
$python3 fen_capture.py
<IMG>
Set the pieces in the starting positions.
Hit 's' to change the location of the white pieces if they are not already in
the correct spot.
Hit 'f' to flip the board view

Power the clock and wait for LEDs to display.
## hit the button on the side of the black pieces.  This starts the white
   clock and triggers an image to be taken of the starting position.
## Make a move with the white pieces and hit the button on the white side 
   of the board.
   <IMG>
## Continue making moves and hitting the clock button.  The moves should
   be visible on the screen.
   <IMG>
## If an error occurs, hit the left arrow key to go back to the last correct
   position as shown on the "Previous Moves" window.  If you go too far, 
   hit the right arrow to re-do moves.  When you are satisifed with the 
   position, hit the button on the clock of the side with the most recently 
   completed move on the board.  The game will resume with the previous time
   on the clock.
   <IMG>
## When the game is over, hit the 'u' key to copy the PGN file and open a 
   web browswer to lichess.org/import and paste (CTRL-V) the game into the
   text input window and click the import button for post-game analysis.
   <IMG>
## hit the 'r' key to reset the board and start a new game.


# Resources
https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
https://www.raspberrypi.com/products/camera-module-v2/
https://www.raspberrypi.com/products/raspberry-pi-touch-display/

https://github.com/karayaman/Play-online-chess-with-real-chess-board
https://pypi.org/project/chess/