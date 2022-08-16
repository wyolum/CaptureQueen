#include <Arduino.h>
#include <TM1637Display.h>

#include <SPIFFS.h>
#include <WiFiSettings.h>

// Module connection pins (Digital Pins)
#define CLK 22
#define DIO 23

#define CLK1 14
#define DIO1 32

TM1637Display display(CLK, DIO);
TM1637Display display1(CLK1, DIO1);

void setup()
{
  Serial.begin(115200);
  SPIFFS.begin(true);  // Will format on the first run after failing to mount

    // Use stored credentials to connect to your WiFi access point.
    // If no credentials are stored or if the access point is out of reach,
    // an access point will be started with a captive portal to configure WiFi.
  WiFiSettings.connect();
}

long counter_seconds = 90L;
void loop()
{
  //7 is the highest brightness
  display1.setBrightness(1);
  display.setBrightness(7);

  display.showNumberDecEx(counter_seconds%60, 0x40, true,2,2);
  display.showNumberDecEx(counter_seconds/60L, 0x40, true,2,0);
  display1.showNumberDecEx(counter_seconds%60, 0x40, true,2,2);
  display1.showNumberDecEx(counter_seconds/60L, 0x40, true,2,0);
 // Serial.print(counter_seconds/60L);Serial.print(":");Serial.println(counter_seconds %60);
  counter_seconds --;
  if (counter_seconds < 0L){
    counter_seconds = 90L;
  }
	delay(100);

}
