import RPi.GPIO as GPIO
import numpy as np
import librosa
import librosa.display
# librosa version == 0.4.2
# and manually modified from 0.7.0
# (logamplitude to same as power_to_db)
# specshow in librosa.display v0.4.2 to v0.7.0
import os
import matplotlib.pyplot as plt
import pyaudio
import time
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from bluetooth import*

class LedControl(object):
    def __init__(self):
        led1_r = 16
        led1_g = 20
        led1_b = 21
        led2_r = 13
        led2_g = 19
        led2_b = 26

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(led1_r, GPIO.OUT)
        GPIO.setup(led1_g, GPIO.OUT)
        GPIO.setup(led1_b, GPIO.OUT)
        GPIO.setup(led2_r, GPIO.OUT)
        GPIO.setup(led2_g, GPIO.OUT)
        GPIO.setup(led2_b, GPIO.OUT)

        hz = 60
        self.pwm1_r = GPIO.PWM(led1_r, hz)
        self.pwm1_g = GPIO.PWM(led1_g, hz)
        self.pwm1_b = GPIO.PWM(led1_b, hz)
        self.pwm2_r = GPIO.PWM(led2_r, hz)
        self.pwm2_g = GPIO.PWM(led2_g, hz)
        self.pwm2_b = GPIO.PWM(led2_b, hz)
        self.pwm1_r.start(100)
        self.pwm1_g.start(100)
        self.pwm1_b.start(100)
        self.pwm2_r.start(100)
        self.pwm2_g.start(100)
        self.pwm2_b.start(100)
        
    def pwm(self, r=None, g=None, b=None):
        if r != None:
            self.pwm1_r.ChangeDutyCycle(r)
            time.sleep(0.001)
            self.pwm2_r.ChangeDutyCycle(r)
            time.sleep(0.001)
        if g != None:
            self.pwm1_g.ChangeDutyCycle(g)
            time.sleep(0.001)
            self.pwm2_g.ChangeDutyCycle(g)
            time.sleep(0.001)
        if b != None:
            self.pwm1_b.ChangeDutyCycle(b)
            time.sleep(0.001)
            self.pwm2_b.ChangeDutyCycle(b)
            time.sleep(0.001)
    
class melCNN(object):
    def __init__(self, sec=3, state=None, label=None, useCNN=None):
        self.FORMAT = pyaudio.paFloat32
        self.SEC = sec
        self.STATE = state
        self.useCNN = useCNN
        
        self.LABEL = label                
        self.labels = {0 : "nothing",
                       1 : "background_noise",
                       2 : "doorbell",
                       3 : "fire_alarm",
                       4 : "hair_dry",
                       5 : "typing",
                       6 : "someone_talking",
                       7 : "baby_crying"}
        
        self.data_dir = os.path.dirname("data/")
        os.makedirs(self.data_dir, exist_ok=True)
        self.train_dir = os.path.join(self.data_dir, "train")
        os.makedirs(self.train_dir, exist_ok=True)
        self.test_dir = os.path.join(self.data_dir, "test")
        os.makedirs(self.test_dir, exist_ok=True)
        self.model_dir = os.path.dirname("model/")
        os.makedirs(self.model_dir, exist_ok=True)
        
        if self.LABEL != None:
            self.label_dir = os.path.join(self.train_dir, self.labels[self.LABEL])
            os.makedirs(self.label_dir, exist_ok=True)
        
        self.CHANNELS = 1
        self.RATE = 44100
        self.n_fft = 1024
        self.hop_length = 1024
        self.n_mels = 128
        self.f_min = 20
        self.f_max = 8000
        
        self.count = 0
        self.data = None
        self.mel = None
        self.total_len = self.RATE * self.SEC
        self.total_data = np.zeros(self.total_len)
        
        dev_idx = None
        self.pa = pyaudio.PyAudio()

        # Detecting microphone.
        for i in range(self.pa.get_device_count()):
            if "USB" in self.pa.get_device_info_by_index(i)['name']:
                dev_idx = self.pa.get_device_info_by_index(i)['index']
                print(dev_idx, self.pa.get_device_info_by_index(i)['name'])
        assert dev_idx != None # Stop if no microphone is detected.

        self.stream = self.pa.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.RATE,
                                   input_device_index=dev_idx,
                                   input=True,
                                   output=False,
                                   frames_per_buffer=self.RATE)
        
    def loop(self):
        try:            
            print("get sound...")
            self.start = time.time()
            self.audioinput()
            
            decibel = round(self.pltmel(),2)
            
            if self.STATE == "test":
                middle = time.time()
                print(str(round(middle-self.start, 3))+"\tsec\tpredicting...")
                pred, acc = self.test(self.useCNN)
                print_str = "dB: "+str(decibel)+"\t"+pred
                
            end = time.time()
            total_time = str(round(end-self.start, 3))+"\tsec\t"
            print(str(self.count)+"\t"+total_time + print_str)
            
            self.count += 1

            return decibel, pred

        except KeyboardInterrupt:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()

            return 0

    def audioinput(self):
        self.data = self.stream.read(self.total_len, exception_on_overflow=False)
        self.total_data = np.fromstring(self.data, np.float32)
            
    def pltmel(self):        
        self.mel = librosa.feature.melspectrogram(y=self.total_data,
                                                  sr=self.RATE,
                                                  n_fft=self.n_fft,
                                                  hop_length=self.hop_length,
                                                  n_mels=self.n_mels,
                                                  #power=1.0,
                                                  fmin=self.f_min,
                                                  fmax=self.f_max)

        # Set image size to w=224, h=224.
        plt.rcParams["figure.figsize"] = (2.24, 2.24)
        plt.axis("off")
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

        self.db_spec = librosa.logamplitude(self.mel, ref=np.max)
        decibel = abs(np.min(self.db_spec))
        lr, out = librosa.display.specshow(self.db_spec, y_axis="mel", x_axis="time")
        # Modified specshow method in display.py to have two outputs.
        # out = out.get_array().reshape(128,130)
        
        if self.STATE == "save_data":
            plt.savefig(str(os.path.join(self.label_dir, "{:03}.jpg".format(self.count))), bbox_inches=None, pad_inches=0, dpi=100)
        elif self.STATE == "test":
            plt.savefig(str(os.path.join(self.test_dir, "{:03}.jpg".format(self.count))), bbox_inches=None, pad_inches=0, dpi=100)
        plt.clf()

        return decibel
        
    def train(self, useCNN=True, epochs=10, hidden=128):
        n_classes = len(self.labels)
        model_name = "model_"+str(n_classes)+"class"        
        data = []
        label = []
        for i in os.listdir(self.train_dir):
            print(i)
            for k,v in self.labels.items():                
                if i == v:                    
                    label_dir = os.path.join(self.train_dir, i)
                    images = os.listdir(label_dir)
                    for img in images:                
                        data.append(plt.imread(os.path.join(label_dir, img)))
                        label.append(k)
        data /= 255.0
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        data, label = shuffle(data, label)
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data, label, test_size=0.3, random_state=0)

        model_name = self.make_model(model_name=model_name, useCNN=useCNN, hidden=hidden, n_classes=n_classes)
        self.train_history = self.model.fit(self.train_x, self.train_y,
                                            epochs=epochs,
                                            validation_data=(self.test_x, self.test_y))
        
        self.model.save_weights(os.path.join(self.model_dir, model_name+".h5"))
        print("Saved model to disk")
        
    def test(self, useCNN=True, hidden=128):        
        if self.count == 0:
            n_classes = len(self.labels)
            model_name = "model_"+str(n_classes)+"class"            
            model_name = self.make_model(model_name=model_name, useCNN=useCNN, hidden=hidden, n_classes=n_classes)
            self.model.load_weights(os.path.join(self.model_dir, model_name+".h5"))
        
        img = plt.imread(os.path.join(self.test_dir, "{:03}.jpg".format(self.count)))
        img = (np.expand_dims(img,0))
        prediction = self.model.predict(img)
        result = np.argmax(prediction[0])
        return self.labels[result], prediction[0][result]
    
    def make_model(self, model_name, n_classes, useCNN=True, hidden=128):        
        if useCNN:
            model_name += "_CNN"
            self.model = keras.Sequential([    
                keras.layers.Conv2D(filters=30, kernel_size=(3,3), activation="relu", padding="valid", input_shape=(224, 224, 3)),
                keras.layers.MaxPooling2D(pool_size=(3,3)),
                keras.layers.Dropout(0.5),
                keras.layers.Conv2D(filters=30, kernel_size=(3, 3), activation="relu", padding="valid"),
                keras.layers.MaxPooling2D(pool_size=(3,3)),
                keras.layers.Flatten(),
                keras.layers.Dense(hidden, activation="relu"),
                keras.layers.Dense(n_classes, activation="softmax")
            ])
        else:
            self.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(224, 224, 3)),
                keras.layers.Dense(hidden, activation="relu"),
                keras.layers.Dense(hidden, activation="relu"),
                keras.layers.Dense(n_classes, activation="softmax")
            ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return model_name

mel_test = melCNN(state="test", useCNN=True)
led = LedControl()
led.pwm(0,0,0)

# Check until Bluetooth connection is established.
while True:
    try:
        client_socket=BluetoothSocket(RFCOMM)
        client_socket.connect(("98:D3:71:FD:5A:F3",1))
        break
    except btcommon.BluetoothError as err:
        print("Error : %s" %err)
        time.sleep(1)

on = True

try: 
    while True:
        start_time = time.time()
        if on:
            led.pwm(r=93,g=92,b=84) # ivory color
        db, pred = mel_test.loop()
        if pred == "doorbell":
            client_socket.send('a')
            for i in range(3):
                led.pwm(r=0,g=100,b=0)
                time.sleep(0.5)
                led.pwm(r=0,g=0,b=0)
                time.sleep(0.5)
        elif pred == "fire_alarm":
            client_socket.send('b')
            for n in range(3):
                for i in range(101):
                    led.pwm(r=i,g=0,b=0)
                    time.sleep(0.01)
                for i in range(101):
                    led.pwm(r=100-i,g=0,b=0)
                    time.sleep(0.01)
        elif pred == "hair_dry":
            client_socket.send('c')
            for n in range(3):
                for i in range(101):
                    led.pwm(r=0,g=0,b=i)
                    time.sleep(0.01)
                for i in range(101):
                    led.pwm(r=0,g=0,b=100-i)
                    time.sleep(0.01)
        elif pred == "baby_crying":
            client_socket.send('d')
            for n in range(3):
                for i in range(101):
                    led.pwm(r=i,g=i,b=0)
                    time.sleep(0.01)
                for i in range(101):
                    led.pwm(r=100-i,g=100-i,b=0)
                    time.sleep(0.01)

except KeyboardInterrupt:
    led.pwm(r=0,g=0,b=0)
    client_socket.close()
