#include <SPI.h>
#include <Wire.h>

#include <Adafruit_SSD1306.h>
#include <SoftwareSerial.h>
SoftwareSerial BT(2, 3); //tx = 2 rx = 3

#define OLED_MOSI   9
#define OLED_CLK   10
#define OLED_DC    11
#define OLED_CS    12
#define OLED_RESET 13
int motor_pin = 4;
char index = 4;
int start_, end_;

Adafruit_SSD1306 display(OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);

void setup() {
  display.begin(SSD1306_SWITCHCAPVCC);
  display.clearDisplay();//화면 초기화
  display.setTextSize(1);//글자 사이즈 설정
  display.setTextColor(WHITE);//글자 색상 설정
  display.setCursor(1, 0); //글자가 표시될 위치 설정
  display.println("Booting...");//출력할 문자 설정
  display.display();//화면에 출력

  display.setCursor(1, 10);

  display.println("...");

  display.display();

  delay(1000);

  display.setCursor(1, 20);

  display.println("Talk Gui Ready!");

  display.display();

  delay(1000);
  BT.begin(9600);
  Serial.begin(9600);
  pinMode(motor_pin, OUTPUT);
  digitalWrite(motor_pin, LOW);
  start_ = millis();
}
void loop() {
  if (BT.available())
  {
    start_ = millis();
    Serial.print("reading:");
    index = BT.read();
    Serial.println(index);
    if (index == 'a') // doorbell
    {
      display.clearDisplay();
      display.setCursor(1, 10);
      display.println("Doorbell Detected");
      display.display();
      digitalWrite(motor_pin, HIGH);
      delay(1000);
      digitalWrite(motor_pin, LOW);
    }
    else if (index == 'b') // Fire ALARM
    {
      display.clearDisplay();
      display.setCursor(1, 10);
      display.println("!!!!!!WARNING!!!!!!");
      display.setCursor(1, 20);
      display.println("Fire ALARM detected");
      display.display();
      digitalWrite(motor_pin, HIGH);
      delay(2000);
      digitalWrite(motor_pin, LOW);
    }
    else if (index == 'c') // Hair dryer
    {
      display.clearDisplay();
      display.setCursor(1, 10);
      display.println("WARNING");
      display.setCursor(1, 20);
      display.println("Hair dryer detected");
      display.display();
      digitalWrite(motor_pin, HIGH);
      delay(1000);
      digitalWrite(motor_pin, LOW);
    }
    else if (index == 'd') // Baby
    {
      display.clearDisplay();
      display.setCursor(10, 10);
      display.println("BABY Crying");
      display.setCursor(10, 20);
      display.println("T.T");
      display.display();
      digitalWrite(motor_pin, HIGH);
      delay(1000);
      digitalWrite(motor_pin, LOW);
    }
  }
  end_ = millis();
  if ( end_ - start_ > 8000)
  {
    display.clearDisplay();
    display.setCursor(1, 15);
    display.println("Everything is fine");
    display.display();
    delay(1000);
  }
}
