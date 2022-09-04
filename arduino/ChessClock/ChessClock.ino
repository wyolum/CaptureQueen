#include <TM1637Display.h>
#include <WiFiManager.h>
#include <PubSubClient.h>

const int CLK=5;
const int DIO=4;
const int CLK1=2;
const int DIO1=16;
long initial_seconds = 5;
long increment_seconds = 5;
long counter_ms[2] = {initial_seconds * 1000, initial_seconds * 1000};
long counter_seconds[2];
int halfmove_number = 0;

const int WHITE = 0;
const int BLACK = 1;

int players[2]; // if {0, 1} player0 is white, player1 is black, else 0-black, 1-white


int turn = 0; // no clock movement to start
bool paused = true; // pause clocks 

TM1637Display display0(CLK, DIO);
TM1637Display display1(CLK1, DIO1);
TM1637Display displays[2] = {display0, display1};

WiFiManager wifiManager;
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

bool bytes2bool(byte* payload, unsigned int length){
  bool out = false;
  if(length > 0){
    out = (char)payload[0] == 't';
  }
  return out;
}

int bytes2int(byte* payload, unsigned int length){
  char str_payload[length + 1];
  for (int i = 0; i < length; i++) {
    str_payload[i] = (char)payload[i];
  }
  str_payload[length] = 0;
  return String(str_payload).toInt();
}

// this is a typedef for topic callback funtions
typedef void (* TopicCallback_p)(byte* payload, unsigned int length);
struct TopicListener{
  char topic[50]; // topic string
  TopicCallback_p callback_p; // action
};

void reset_cb(byte *payload, unsigned int length){
  String str_temp;
  new_game(false);
}

void pause_cb(byte *payload, unsigned int length){
  paused = bytes2int(payload, length);
}

void setturn_cb(byte *payload, unsigned int length){
  turn = bytes2int(payload, length);
}

void sethalfmove_number_cb(byte *payload, unsigned int length){
  halfmove_number = bytes2int(payload, length);
}

void setwhite_ms_cb(byte *payload, unsigned int length){
  counter_ms[players[WHITE]] = bytes2int(payload, length);
}

void setblack_ms_cb(byte *payload, unsigned int length){
  counter_ms[players[BLACK]] = bytes2int(payload, length);
}

void set_initial_seconds_cb(byte *payload, unsigned int length){
  initial_seconds = bytes2int(payload, length);
}

void set_increment_cb(byte *payload, unsigned int length){
  increment_seconds = bytes2int(payload, length);
}
TopicListener reset_listener = {"capture_queen.reset", reset_cb};
TopicListener pause_listener = {"capture_queen.paused", pause_cb};
TopicListener setturn_listener = {"capture_queen.setturn", setturn_cb};
TopicListener sethalfmove_number_listener = {
  "capture_queen.sethalf_move_mumber",
  sethalfmove_number_cb};
TopicListener setwhite_ms_listener = {"capture_queen.setwhite_ms",
				      setwhite_ms_cb};
TopicListener setblack_ms_listener = {"capture_queen.setblack_ms",
				      setblack_ms_cb};
TopicListener increment_listener = {"capture_queen.increment_seconds",
				    set_increment_cb};
TopicListener initial_seconds_listener = {"capture_queen.initial_seconds",
					  set_initial_seconds_cb};

const int N_TOPIC_LISTENERS = 8;
TopicListener *TopicListeners[N_TOPIC_LISTENERS] = {
  &reset_listener,
  &pause_listener,
  &setturn_listener,
  &increment_listener,
  &initial_seconds_listener,
  &sethalfmove_number_listener,
  &setblack_ms_listener,
  &setwhite_ms_listener
};

void setup_wifi() {
  //wifiManager.resetSettings(); // uncomment to forget network settings
  delay(10);
  // We start by connecting to a WiFi network
  // reset network?
  wifiManager.autoConnect("CaptureQueen");

  Serial.println("Yay connected!");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

}

void subscribe(){
  for(int i=0; i < N_TOPIC_LISTENERS; i++){
    mqtt_client.subscribe(TopicListeners[i]->topic);
    Serial.print("Subscribted to ");
    Serial.println(TopicListeners[i]->topic);
  }
}

void publish_msg(char* topic, const char* msg){
  Serial.print("publish_msg::");
  Serial.print(topic);
  Serial.print("::");
  Serial.println(msg);
  mqtt_client.publish(topic, msg);
}

void publish_int(char* topic, int val){
  String val_str = String(val);
  publish_msg(topic, val_str.c_str());
}

void mqtt_publish_state(){
  String msg = String(players[turn]) + String("//") +
    String(counter_ms[players[WHITE]]) + String("//") + 
    String(counter_ms[players[BLACK]]);
  publish_msg("capture_queen.turn",  msg.c_str());
}
void mqtt_connect(){
  String str;
  
  while (!mqtt_client.connected()) {
    if(mqtt_client.connect("ESP32Client")) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      // ... and resubscribe
      subscribe();
    }
    else{
      Serial.print("Try again in 5 seconds.");
      delay(5000);
    }
  }
}

void mqtt_reconnect() {
  // Loop until we're reconnected
  while (!mqtt_client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (mqtt_client.connect("ESP32Client")) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      // ... and resubscribe
      subscribe();
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqtt_client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void display_setup(){
  display0.setBrightness(7);   //7 is the highest brightness
  display1.setBrightness(3);   //7 is the highest brightness

}

uint8_t ZEROS[] = {0, 0, 0, 0};
void display_loop(){
  counter_seconds[0] = counter_ms[0] / 1000;
  counter_seconds[1] = counter_ms[1] / 1000;

  if(counter_ms[0] >= 10000){
    if(counter_seconds[0] >= 3600){ // display hh:mm
      int colen = (millis() % 1000 < 500) * 0x40;
      if(turn != 0){ // keep other colen steady on
	colen = 0x40;
      }
      int hh = counter_seconds[0] / 3600;
      int mm = (counter_seconds[0] / 60) % 60;
      display0.showNumberDecEx(hh, colen, false, 2, 0);
      display0.showNumberDecEx(mm, colen, true, 2, 2);
    }
    else{
      display0.showNumberDecEx(counter_seconds[0]  % 60, 0x40, true, 2, 2);
      display0.showNumberDecEx(counter_seconds[0] / 60, 0x40, true, 2, 0);
    }
  }
  else{
    display0.setSegments(ZEROS, 1, 0);
    display0.setSegments(ZEROS, 1, 3);
    display0.showNumberDecEx((counter_ms[0] / 100), 0xFF, true, 2, 1);
  }
  if(counter_ms[1] >= 10000){
    if(counter_seconds[1] >= 3600){ // display hh:mm
      int colen = (millis() % 1000 < 500) * 0x40;
      if(turn != 1){ // keep other colen steady on
	colen = 0x40;
      }
      int hh = counter_seconds[1] / 3600;
      int mm = (counter_seconds[1] / 60) % 60;
      display1.showNumberDecEx(hh, colen, false, 2, 0);
      display1.showNumberDecEx(mm, colen, true, 2, 2);
    }
    else{
      display1.showNumberDecEx(counter_seconds[1] % 60, 0x40, true, 2, 2);
      display1.showNumberDecEx(counter_seconds[1] / 60, 0x40, true, 2, 0);
    }
  }
  else{
    display1.setSegments(ZEROS, 1, 0);
    display1.setSegments(ZEROS, 1, 3);
    display1.showNumberDecEx((counter_ms[1] / 100), 0xFF, true, 2, 1);
  }
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  bool handled = false;

  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.println("] ");
  for(int i=0; i < N_TOPIC_LISTENERS && !handled; i++){
    if(strcmp(topic, TopicListeners[i]->topic) == 0){
      TopicListeners[i]->callback_p(payload, length);
      handled = true;
    }
  }
  if(!handled){
    for (int i = 0; i < length; i++) {
      Serial.print(payload[i]);
    }
    Serial.println();
    Serial.println("Not handled.");
  }
}

void setup_mqtt(){
  uint8_t mqtt_server[4] = {192, 168, 7, 130};
  mqtt_client.setServer(mqtt_server, 1883);
  mqtt_client.setCallback(mqtt_callback);
  mqtt_connect();
}
const int player_0_pin = 14;
const int player_1_pin = 0;


void setup(){
  Serial.begin(115200);delay(10);
  Serial.println("\n\n\nCapture Queen, open hardware.\n\n\n");
  setup_wifi();
  setup_mqtt();
  
  
  pinMode(player_0_pin, INPUT_PULLUP);
  pinMode(player_1_pin, INPUT_PULLUP);
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
  display_setup();
  new_game(false);
}

bool check_for_reset(){
  bool out = false;
  if(!(digitalRead(player_0_pin) || digitalRead(player_1_pin))){
    delay(250); // longer press than accident
    out = !(digitalRead(player_0_pin) || digitalRead(player_1_pin));
    if(out){
      display0.showNumberDecEx(8888, 0xFF, true, 4, 0);
      display1.showNumberDecEx(8888, 0xFF, true, 4, 0);
      while(!digitalRead(player_0_pin) || !digitalRead(player_1_pin)){
	delay(1);
      }
    }
  }
  return out;
}

void new_game(bool reset_pi){
  turn = 0;
  halfmove_number = 0;
  paused = true;
  for(int i = 0; i < 2; i++){
    counter_ms[i] = initial_seconds * 1000;
  }
  display0.clear();
  display1.clear();
  delay(100);
  mqtt_publish_state();
  if(reset_pi){
    publish_int("capture_queen.reset_pi",  3);
  }
}

void game_over(int player){
  mqtt_publish_state();
  Serial.print("Time ran out for");
  Serial.println(player);
  Serial.println(counter_ms[player] / 1000);
  Serial.println();
  if(player == 0){
    display0.showNumberDecEx(0, 0xFF, true, 4, 0);
    while(!check_for_reset()){
      delay(1);
    }
  }
  else{
    display1.showNumberDecEx(0, 0xFF, true, 4, 0);
    while(!check_for_reset()){
      delay(1);
    }
  }
  new_game(false);
}

int buttons[2] = {player_0_pin, player_1_pin};
int last_time_ms = 0;
void loop(){
  int button_state[2];

  if (!mqtt_client.connected()) {
    mqtt_reconnect();
  }

  
  if(paused){ // game paused... unpause game, start other timer
    button_state[0] = digitalRead(player_0_pin);
    button_state[1] = digitalRead(player_1_pin);
    if(halfmove_number > 0){ // only check one button for current player
      turn = halfmove_number % 2;
      if(button_state[turn] == 0){
	paused = false;
      }
    }
    else{
      if(button_state[0] == 0){
	paused = false;
	players[0] = BLACK; // player 0 is black
	players[1] = WHITE; // player 1 is white
	Serial.println("Game on!");
	turn = 1;
	last_time_ms = millis();
	mqtt_publish_state();
	//counter_ms[players[turn]] -= increment_seconds * 1000;
      }
      else if(button_state[1] == 0){
	paused = false;
	players[0] = WHITE; // player 0 is white
	players[1] = BLACK; // player 1 is black
	Serial.println("Game on!");
	turn = 0;
	last_time_ms = millis();
	mqtt_publish_state();
	// pre-subtract off increment.. will get added when move starts
	//counter_ms[players[turn]] -= increment_seconds * 1000;
      }
    }
  }
  if(!paused){// can't use 'else' since players can unpause game above
    halfmove_number++;
    if(digitalRead(buttons[turn]) == 0){
      if(halfmove_number > 1){
	counter_ms[turn] += increment_seconds * 1000;
      }
      
      turn += 1;
      turn %= 2;
      mqtt_publish_state();      
    }
    int now_ms = millis();
    int delta_ms = (now_ms - last_time_ms) * (1 - paused);
    counter_ms[turn]-= delta_ms;
    last_time_ms = now_ms;
  }
  if (counter_ms[0] < 0){
    game_over(0);
  }
  if (counter_ms[1] < 0){
    game_over(1);
  }

  if(check_for_reset()){
    new_game(true);
  }
  display_loop();
  mqtt_client.loop();
  
}
