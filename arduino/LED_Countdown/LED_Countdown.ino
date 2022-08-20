#include <Arduino.h>
#include <TM1637Display.h>


// Module connection pins (Digital Pins)
//#define CLK 22
//#define DIO 23
//#define CLK1 14
//#define DIO1 32
#define CLK 5
#define DIO 4
#define CLK1 2
#define DIO1 16


TM1637Display display0(CLK, DIO);
TM1637Display display1(CLK1, DIO1);

void setup()
{
  Serial.begin(115200);
  //7 is the highest brightness
  display0.setBrightness(1);
  display1.setBrightness(3);

    // Use stored credentials to connect to your WiFi access point.
    // If no credentials are stored or if the access point is out of reach,
    // an access point will be started with a captive portal to configure WiFi.
  //WiFiSettings.connect();
}

long counter_seconds0 = 90L;
long counter_seconds1 = 234L;
void loop()
{

  display0.showNumberDecEx(counter_seconds0%60, 0x40, true,2,2);
  display0.showNumberDecEx(counter_seconds0/60L, 0x40, true,2,0);
  display1.showNumberDecEx(counter_seconds1%60, 0x40, true,2,2);
  display1.showNumberDecEx(counter_seconds1/60L, 0x40, true,2,0);
 // Serial.print(counter_seconds/60L);Serial.print(":");Serial.println(counter_seconds %60);
  counter_seconds0 --;
  counter_seconds1 --;
  if (counter_seconds0 < 0L){
    counter_seconds0 = 90L;
  }
  if (counter_seconds1 < 0L){
    counter_seconds1 = 90L;
  }
	delay(100);

}
