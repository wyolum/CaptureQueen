const int player_1 = 14;
const int player_2 = 0;

void setup(){
  pinMode(player_1, INPUT_PULLUP);
  pinMode(player_2, INPUT_PULLUP);
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
  Serial.begin(115200);
}

void loop(){
  Serial.print(digitalRead(player_1));
  Serial.println(digitalRead(player_2));
  delay(250);
}
