#include <TM1637Display.h>

const int CLK=5;
const int DIO=4;
const int CLK1=2;
const int DIO1=16;

int turn = 3; // no clock movement to start

TM1637Display display0(CLK, DIO);
TM1637Display display1(CLK1, DIO1);
TM1637Display displays[2] = {display0, display1};

void display_setup(){
  display0.setBrightness(1);   //7 is the highest brightness
  display1.setBrightness(3);   //7 is the highest brightness

}
long initial_seconds = 11;
long counter_ms[2] = {11000, 11000};
long counter_seconds[2];

bool ten_second_started[2] = {false, false};
void display_loop(){
  counter_seconds[0] = counter_ms[0] / 1000;
  counter_seconds[1] = counter_ms[1] / 1000;
  if(counter_ms[0] >= 10000){
    display0.showNumberDecEx(counter_seconds[0]  % 60, 0x40, true, 2, 2);
    display0.showNumberDecEx(counter_seconds[0] / 60, 0x40, true, 2, 0);
  }
  else{
    if(!ten_second_started[0]){
      display0.clear();
      ten_second_started[0] = true;
    }
    display0.showNumberDecEx((counter_ms[0] / 100), 0xFF, true, 2, 1);
  }
  if(counter_ms[1] >= 10000){
    display1.showNumberDecEx(counter_seconds[1] % 60, 0x40, true, 2, 2);
    display1.showNumberDecEx(counter_seconds[1] / 60, 0x40, true, 2, 0);
  }
  else{
    if(!ten_second_started[1]){
      display1.clear();
      ten_second_started[1] = true;
    }
    display1.showNumberDecEx((counter_ms[1] / 100), 0xFF, true, 2, 1);
  }
}

const int player_0_pin = 14;
const int player_1_pin = 0;

void setup(){
  pinMode(player_0_pin, INPUT_PULLUP);
  pinMode(player_1_pin, INPUT_PULLUP);
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
  Serial.begin(115200);
  display_setup();
  Serial.println("Capture Queen, open hardware");
  new_game();
}

bool reset(){
  bool out = false;
  if(!(digitalRead(player_0_pin) || digitalRead(player_1_pin))){
    delay(250); // longer press than accident
    out = !(digitalRead(player_0_pin) || digitalRead(player_1_pin));
  }
  return out;
}

void new_game(){
  turn = 3;
  for(int i = 0; i < 2; i++){
    counter_ms[i] = initial_seconds * 1000;
    ten_second_started[i] = false;
  }
  display0.showNumberDecEx(8888, 0xFF, true, 4, 0);
  display1.showNumberDecEx(8888, 0xFF, true, 4, 0);
  delay(1000);
  display0.clear();
  display1.clear();
  delay(100);

}

void game_over(int player){
  if(player == 0){
    display0.showNumberDecEx(0, 0xFF, true, 4, 0);
    while(!reset()){
      delay(1);
    }
  }
  else{
    display1.showNumberDecEx(0, 0xFF, true, 4, 0);
    while(!reset()){
      delay(1);
    }
  }
  new_game();
}

int buttons[2] = {player_0_pin, player_1_pin};
int last_time_ms = 0;
void loop(){
  int button_state[2];
  if(turn > 2){ // game not started
    button_state[0] = digitalRead(player_0_pin);
    button_state[1] = digitalRead(player_1_pin);
    if(button_state[0] == 0){
      Serial.println("Game on!");
      turn = 1;
      last_time_ms = millis();
    }
    else if(button_state[1] == 0){
      Serial.println("Game on!");
      turn = 0;
      last_time_ms = millis();
    }
  }
  else{
    if(digitalRead(buttons[turn]) == 0){
      turn += 1;
      turn %= 2;
    }
    int now_ms = millis();
    int delta_ms = now_ms - last_time_ms;
    counter_ms[turn]-= delta_ms;
    last_time_ms = now_ms;
  }
  if (counter_ms[0] < 0){
    game_over(0);
  }
  if (counter_ms[1] < 0){
    game_over(1);
  }

  if(reset()){
    new_game();
  }
  display_loop();
}
