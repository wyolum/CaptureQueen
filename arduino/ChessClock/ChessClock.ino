#include <TM1637Display.h>
#include <WiFiManager.h>
#include <PubSubClient.h>
#include <ESP8266HTTPClient.h>

const int CLK=5;
const int DIO=4;
const int CLK1=2;
const int DIO1=16;
//long initial_seconds = 300;
long initial_seconds = 300;
long increment_seconds = 0;
long counter_ms[2] = {initial_seconds * 1000, initial_seconds * 1000};
long counter_seconds[2];
int halfmove_number = 0;
int last_interaction_ms = 0;

const int WHITE = 0;
const int BLACK = 1;
const int player_0_pin = 14;
const int player_1_pin = 0;

int players[2]; // if {0, 1} player0 is white, player1 is black, else 0-black, 1-white


int turn = 0; // no clock movement to start
bool paused = true; // pause clocks 

TM1637Display display0(CLK, DIO);
TM1637Display display1(CLK1, DIO1);
TM1637Display displays[2] = {display0, display1};

WiFiManager wifiManager;
WiFiClient espClient;
PubSubClient mqtt_client(espClient);

void parse_ip(String s, uint8_t* int4){
  int start = 0;
  int stop;
  String substr;
  
  for(int i=0; i<4; i++){
    stop = s.indexOf('.', start);
    substr = s.substring(start, stop);
    int4[i] = (uint8_t)substr.toInt();
    start = stop + 1;
  }
}

String jsonLookup(String s, String name){
  int start = s.indexOf(name) + name.length() + 3;
  int stop = s.indexOf('"', start);
  //Serial.println(s.substring(start, stop));
  return s.substring(start, stop);
}

uint8_t mqtt_ip_address[4] = {0, 0, 0, 0};
void get_mqtt_server_ip(){

  HTTPClient http;
  
  Serial.print("[HTTP] begin...\n");
  //String url = String("http://www.wyolum.com/utc_offset/get_localips.py") +
  //  String("?dev_type=CaptureQueen.Mosquitto");
  String url = String("http://wyolum.com/utc_offset/utc_offset.py") +
    String("?dev_type=CaptureQueen.Clock") +
    String("&localip=") +
    String(WiFi.localIP()[0]) + String('.') + 
    String(WiFi.localIP()[1]) + String('.') + 
    String(WiFi.localIP()[2]) + String('.') + 
    String(WiFi.localIP()[3]) + String('&') +
    String("macaddress=") + WiFi.macAddress();

  Serial.println(url);
  http.begin(url);
  
  Serial.print("[HTTP] GET...\n");
  // start connection and send HTTP header
  int httpCode = http.GET();
  Serial.printf("[HTTP] ... code: %d\n", httpCode);
  
  // httpCode will be negative on error
  if(httpCode > 0) {
    // HTTP header has been send and Server response header has been handled
    Serial.printf("[HTTP] GET... code: %d\n", httpCode);
    // file found at server
    //String findme = String("offset_seconds");
    if(httpCode == HTTP_CODE_OK) {
      String payload = http.getString();
      Serial.print("payload:");
      Serial.println(payload);
      payload.replace(" ", "");
      String mqtt_ip_str = jsonLookup(payload, String("mqtt_ip_address"));
      parse_ip(mqtt_ip_str, mqtt_ip_address);
      for(int ii=0; ii<4; ii++){
	Serial.print(mqtt_ip_address[ii]);
	if(ii < 3){
	  Serial.print(".");
	}
      }
      Serial.println();
    }
    else{
      Serial.println("No mqtt service found");
    }
  }
}


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
  new_game();
}

void game_over_cb(byte *payload, unsigned int length){
  String str_temp;
  game_over();
}

void pause_cb(byte *payload, unsigned int length){
  paused = bytes2int(payload, length);
}

void setturn_cb(byte *payload, unsigned int length){
  turn = bytes2int(payload, length);
}

void sethalfmove_cb(byte *payload, unsigned int length){
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
TopicListener game_over_listener = {"capture_queen.game_over", game_over_cb};
TopicListener reset_listener = {"capture_queen.reset", reset_cb};
TopicListener pause_listener = {"capture_queen.paused", pause_cb};
TopicListener setturn_listener = {"capture_queen.setturn", setturn_cb};
TopicListener sethalfmove_listener = {
  "capture_queen.sethalfmove", sethalfmove_cb};
TopicListener setwhite_ms_listener = {"capture_queen.setwhite_ms",
				      setwhite_ms_cb};
TopicListener setblack_ms_listener = {"capture_queen.setblack_ms",
				      setblack_ms_cb};
TopicListener increment_listener = {"capture_queen.increment_seconds",
				    set_increment_cb};
TopicListener initial_seconds_listener = {"capture_queen.initial_seconds",
					  set_initial_seconds_cb};

const int N_TOPIC_LISTENERS = 9;
TopicListener *TopicListeners[N_TOPIC_LISTENERS] = {
  &game_over_listener,
  &reset_listener,
  &pause_listener,
  &setturn_listener,
  &increment_listener,
  &initial_seconds_listener,
  &sethalfmove_listener,
  &setblack_ms_listener,
  &setwhite_ms_listener
};

void setup_wifi() {
  //wifiManager.resetSettings(); // uncomment to forget network settings
  delay(10);
  // We start by connecting to a WiFi network
  // reset network?
  if(!digitalRead(player_0_pin) && !digitalRead(player_1_pin)){
    wifiManager.resetSettings();
  }
    
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
bool mqtt_connect(){
  String str;
  int n_try = 0;
  int max_tries = 3;
  
  while (!mqtt_client.connected() && n_try < max_tries) {
    if(mqtt_client.connect("ChessClock")) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      // ... and resubscribe
      subscribe();
    }
    else{
      n_try++;
      Serial.print("Try again in 5 seconds. ");
      Serial.print(n_try);
      Serial.print("/");
      Serial.println(max_tries);
      delay(5000);
    }
  }
}

void mqtt_reconnect() {
  // Loop until we're reconnected
  while (!mqtt_client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (mqtt_client.connect("ChessClock")) {
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
void set_display(int display_num, TM1637Display display, int counter_ms){
  int counter_seconds;
  
  counter_seconds = counter_ms / 1000;
  
  if(counter_ms >= 60000){ // longer than 10 minutes display mm:ss
    if(counter_seconds >= 3600){ // longer than an hour display hh:mm
      int colen = (millis() % 1000 < 500) * 0x40;
      bool myturn = (display_num == turn);
      if(!myturn or paused){ // keep other colen steady on
	colen = 0x40;
      }
      
      int hh = counter_seconds / 3600;
      int mm = (counter_seconds / 60) % 60;
      display.showNumberDecEx(hh, colen, false, 2, 0);
      display.showNumberDecEx(mm, colen, true, 2, 2);
    }
    else{
      display.showNumberDecEx(counter_seconds  % 60, 0x40, true, 2, 2);
      display.showNumberDecEx(counter_seconds / 60, 0x40, true, 2, 0);
    }
  }
  else{
    display.setSegments(ZEROS, 1, 3);
    display.showNumberDecEx((counter_ms / 100), 0xFF, false, 3, 0);
  }
}
void display_loop(){
  set_display(0, display0, counter_ms[0]);
  set_display(1, display1, counter_ms[1]);
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  bool handled = false;

  last_interaction_ms = millis();

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
  get_mqtt_server_ip();
  //uint8_t mqtt_server[4] = {192, 168, 7, 130};
  mqtt_client.setServer(mqtt_ip_address, 1883);
  mqtt_client.setCallback(mqtt_callback);
  mqtt_connect();
}

void button_setup(){
  pinMode(player_0_pin, INPUT_PULLUP);
  pinMode(player_1_pin, INPUT_PULLUP);
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
}
void setup(){
  button_setup();
  Serial.begin(115200);delay(10);
  Serial.println("\n\n\nCapture Queen, open hardware.\n\n\n");
  setup_wifi();
  setup_mqtt();
  
  
  display_setup();
  new_game();
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

void new_game(){
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
  publish_int("capture_queen.reset_pi",  3);
}

void game_over(){
  mqtt_publish_state();
  if(counter_ms[0] < 0){
    display0.showNumberDecEx(0, 0xFF, true, 4, 0);
  }
  if(counter_ms[1] < 0){
    display1.showNumberDecEx(0, 0xFF, true, 4, 0);
  }
  while(!check_for_reset()){
    delay(1);
  }
  new_game();
}

const int TIMEOUT_MS = 5 * 60000;
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
    if(button_state[0] == 0 || button_state[1] == 0){
      last_interaction_ms = millis();
    }
    if(halfmove_number > 0){ // only check one button for current player
      turn = halfmove_number % 2;
      if(button_state[turn] == 0){
	paused = false;
	last_time_ms = millis();
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
    if(initial_seconds > 0){
      int now_ms = millis();
      int delta_ms = (now_ms - last_time_ms) * (1 - paused);
      counter_ms[turn]-= delta_ms;
      last_time_ms = now_ms;
    }
  }
  if (counter_ms[0] < 0){
    game_over();
  }
  if (counter_ms[1] < 0){
    game_over();
  }

  if(check_for_reset()){
    new_game();
  }
  display_loop();
  mqtt_client.loop();
  if(paused && millis() - last_interaction_ms > TIMEOUT_MS){
    sleep();
  }
}

void sleep(){
  bool asleep = true;
  display0.clear();
  display1.clear();
  Serial.println("Going to sleep");
  while(asleep){
    if(digitalRead(player_0_pin) == 0 ||
       digitalRead(player_1_pin) == 0){
      asleep = false;
      display_loop();
      while(digitalRead(player_0_pin) == 0 ||
	    digitalRead(player_1_pin) == 0){
	delay(100); // don't go on untill button is no longer pressed
      }
    }
    delay(100);
  }
  Serial.println("Waking up");
  last_interaction_ms = millis();
}
