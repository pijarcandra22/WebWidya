from flask import Flask, render_template, request, url_for, redirect,session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
import tensorflow as tf
from tensorflow import keras
import librosa
import librosa.display
import numpy as np
import uuid
import os
import pandas as pd

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

scaler = StandardScaler()
scaler.fit(pd.read_csv("7Preprocces.csv").iloc[:,:-1].values)
model = keras.models.load_model("py/model.h5")

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_audio(directory,name_file):
  if name_file not in request.files:
    return False
  file = request.files[name_file]
  if file.filename == '':
    return False
  if file and allowed_file(file.filename):
    app.config['UPLOAD_FOLDER'] = directory
    filename = secure_filename(file.filename)
    formatfile=filename.split('.')
    newfilename=str(uuid.uuid4().hex)+'.'+formatfile[1]
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    os.rename(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.path.join(app.config['UPLOAD_FOLDER'], newfilename))
    return newfilename

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
  audio=upload_audio("data","audio")
  if not audio:
    return "None"
  
  feature = get_features("data/"+audio)
  X = []
  for ele in feature:
    X.append(ele)
  
  X = np.array(X)
  X = scaler.transform(X)
  predict = model.predict(X)
  bigpredict = [max(x) for x in predict]
  bestpredict = predict[bigpredict.index(max(bigpredict))].tolist()
  hasil = bestpredict.index(max(bestpredict))
  return str(hasil)

def noise(data): #Deklarasi Fungsi
    noise_amp = 0.04*np.random.uniform()*np.amax(data)#Mencari nilai random untuk noise
    data = data + noise_amp*np.random.normal(size=data.shape[0]) #Menambahkan nilai random pada noise
    return data #mengembalikan hasil

def stretch(data, rate=0.70): #Mengatur 
    return librosa.effects.time_stretch(data, rate=rate)
    
def higher_speed(data, speed_factor = 1.25):
    return librosa.effects.time_stretch(data, rate=speed_factor)

def lower_speed(data, speed_factor = 0.75):
    return librosa.effects.time_stretch(data, rate=speed_factor)

def pitch(data, sampling_rate, pitch_factor=0.8):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def extract_features(data):
    
    result = np.array([]) #Menyiapkan Array KOSONG
    
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58) #Mengekstrak MFCC dari data audio dengan sample rate sebesar 22050
    #fungsi mfcc berjalan dengan mengubah data input audio menjadi data numerik berdasar pada sample rate dan jumlah fitur mfcc yang diperlukan
    #Selanjutnya karena hasil fitur mfcc berupa data 2 dimensi maka perlu dicari nilai rata rat untuk tiap kolom dari mfcc yang dihasilkan
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
     
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=3, offset=0.5, res_type='kaiser_fast') 
    
    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    #noised
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically
    
    #stretched
    stretch_data = stretch(data)
    res3 = extract_features(stretch_data)
    result = np.vstack((result, res3))
    
    #shifted
    shift_data = shift(data)
    res4 = extract_features(shift_data)
    result = np.vstack((result, res4))
    
    #pitched
    pitch_data = pitch(data, sample_rate)
    res5 = extract_features(pitch_data)
    result = np.vstack((result, res5)) 
    
    #speed up
    higher_speed_data = higher_speed(data)
    res6 = extract_features(higher_speed_data)
    result = np.vstack((result, res6))
    
    #speed down
    lower_speed_data = higher_speed(data)
    res7 = extract_features(lower_speed_data)
    result = np.vstack((result, res7))
    
    return result

if __name__=='__main__':
  app.run()