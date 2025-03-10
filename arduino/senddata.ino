// Arduino sketch for reading MPU6050 and buttons

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// MPU6050 sensor object
Adafruit_MPU6050 mpu;

// Button pins
const int BUTTON1_PIN = 3;
const int BUTTON2_PIN = 4;
const int BUTTON3_PIN = 5;


void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Set button pins as inputs with internal pull-up resistors
  pinMode(BUTTON1_PIN, INPUT_PULLUP);
  pinMode(BUTTON2_PIN, INPUT_PULLUP);
  pinMode(BUTTON3_PIN, INPUT_PULLUP);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize MPU6050
  Serial.println("Initializing MPU6050...");
  if (!mpu.begin()) {
      Serial.println("Failed to find MPU6050 chip");
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("");

  delay(100);
}

void loop() {
  // Read button states (invert since pull-up resistors make pressed=LOW)
  int but1 = !digitalRead(BUTTON1_PIN);
  int but2 = !digitalRead(BUTTON2_PIN);
  int but3 = !digitalRead(BUTTON3_PIN);
  
  // Read MPU6050 sensor data
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // Scale values (optional - you can adjust scaling as needed)
  float accel_x = a.acceleration.x / 16384.0;  // For ±2g range
  float accel_y = a.acceleration.y / 16384.0;
  float accel_z = a.acceleration.z / 16384.0;
  
  float gyro_x = g.gyro.x;
  float gyro_y = g.gyro.y;
  float gyro_z = g.gyro.z;
  
  // Send data over serial as comma-separated values
  Serial.print(but1);
  Serial.print(",");
  Serial.print(but2);
  Serial.print(",");
  Serial.print(but3);
  Serial.print(",");
  Serial.print(accel_x, 4);
  Serial.print(",");
  Serial.print(accel_y, 4);
  Serial.print(",");
  Serial.print(accel_z, 4);
  Serial.print(",");
  Serial.print(gyro_x);
  Serial.print(",");
  Serial.print(gyro_y);
  Serial.print(",");
  Serial.println(gyro_z);
  
  // Short delay between readings
  delay(50);
}
