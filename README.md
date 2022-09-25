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

## Set up network
1. Start wifi hotspot (or use local wireless area network WAN)
1. Connect pi to hotspot
1. Start fen_capture.py on pi
1. Power clock
1. find access point CaptureQueen (to get clock on line)
1. log clock onto WAN
1. reset fen_capture.py (type 'r')

Power the clock and wait for LEDs to display.
1. hit the button on the side of the black pieces.  This starts the white
   clock and triggers an image to be taken of the starting position.
1. Make a move with the white pieces and hit the button on the white side 
   of the board.
   <IMG>
1. Continue making moves and hitting the clock button.  The moves should
   be visible on the screen.
   <IMG>
1. If an error occurs, hit the left arrow key to go back to the last correct
   position as shown on the "Previous Moves" window.  If you go too far, 
   hit the right arrow to re-do moves.  When you are satisifed with the 
   position, hit the button on the clock of the side with the most recently 
   completed move on the board.  The game will resume with the previous time
   on the clock.
   <IMG>
1. When the game is over, hit the 'u' key to copy the PGN file and open a 
   web browswer to lichess.org/import and paste (CTRL-V) the game into the
   text input window and click the import button for post-game analysis.
   <IMG>
1. hit the 'r' key to reset the board and start a new game.

1. (Optional) On laptop, kick-off the mqtt_board_client.py with the ip 
  address of the mosquitto server. Open(.board.svg) in eog (eye of gnome) to 
  see current images of the game in progress.

# Resources
1. https://www.raspberrypi.com/products/raspberry-pi-4-model-b/
1. https://www.raspberrypi.com/products/camera-module-v2/
1. https://www.raspberrypi.com/products/raspberry-pi-touch-display/
1. https://www.adafruit.com/product/3671

1. https://github.com/karayaman/Play-online-chess-with-real-chess-board
1. https://pypi.org/project/chess/
1. https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
1. https://www.bogotobogo.com/python/python-Windows-Check-if-a-Process-is-Running-Hanging-Schtasks-Run-Stop.php