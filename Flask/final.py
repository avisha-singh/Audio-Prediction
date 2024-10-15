# IMPORTING LIBRARIES
# --------------------------------------------->>>
import pandas as pd
from pandas import DataFrame
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import librosa
from flask import Flask,render_template,request

ALLOWED_EXTENSIONS = set(['wav'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def web():
    return render_template('Page.html')

@app.route("/predict",methods=['post'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):

            # LOADING PICKLE MODULES
            # ----------------------------------------------------------------------------->>>
            loaded_model = pickle.load(open('/IC272/PYTHON/PROJECT/Flask/models/new/model.pkl', 'rb'))
            min_value = pickle.load(open('/IC272/PYTHON/PROJECT/Flask/models/new/min.pkl', 'rb'))
            max_value = pickle.load(open('/IC272/PYTHON/PROJECT/Flask/models/new/max.pkl', 'rb'))
            Data = pickle.load(open('/IC272/PYTHON/PROJECT/Flask/models/new/data.pkl', 'rb')) 

            # READING AUDIO FILE
            # ------------------------------------------------------------------------------>>>
            y, sr = librosa.load(file,duration=3)
            # CREATING A LIST OF FEATURES
            features=[]

            # EXTRACTING FEATURES
            # ------------------------------------------------------------------------------->>>
            # CHROMA_STFT (MEAN AND VARIANCE)
            chrom=librosa.feature.chroma_stft(y=y, sr=sr)
            # Finding mean and variance
            chrom_mean=chrom.mean()
            chrom_var=chrom.var()
            # Appending
            features.append(chrom_mean)
            features.append(chrom_var)

            # RMS (MEAN AND VARIANCE)
            rms=librosa.feature.rms(y=y)
            # Finding mean and variance
            rms_mean=rms.mean()
            rms_var=rms.var()
            # Appending
            features.append(rms_mean)
            features.append(rms_var)

            # SPECTRAL CENTROID (MEAN AND VARIANCE)
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            # Finding mean and variance
            cent_mean=cent.mean()
            cent_var=cent.var()
            # Appending
            features.append(cent_mean)
            features.append(cent_var)

            # SPECTRAL BANDWIDTH (MEAN AND VARIANCE)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            # Finding mean and variance
            spec_bw_mean=spec_bw.mean()
            spec_bw_var=spec_bw.var()
            # Appending
            features.append(spec_bw_mean)
            features.append(spec_bw_var)

            # ROLL OFF (MEAN AND VARIANCE)
            roll=librosa.feature.spectral_rolloff(y=y, sr=sr)
            # Finding mean and variance
            roll_mean=roll.mean()
            roll_var=roll.var()
            # Appending
            features.append(roll_mean)
            features.append(roll_var)

            # ZERO CROSSING RATE (MEAN AND VARIANCE)
            zcr=librosa.feature.zero_crossing_rate(y)
            # Finding mean and variance
            zcr_mean=zcr.mean()
            zcr_var=zcr.var()
            # Appending
            features.append(zcr_mean)
            features.append(zcr_var)

            # HARMONY AND PERCEPTR (MEAN AND VARIANCE)
            trim_y,_= librosa.effects.trim(y)
            y_harmonic, y_percep = librosa.effects.hpss(trim_y)
            # Finding mean and variance of Harmony
            harmony_mean=y_harmonic.mean()
            harmony_var=y_harmonic.var()
            # Appending
            features.append(harmony_mean)
            features.append(harmony_var)
            # Finding mean and variance of Perceptr
            perceptr_mean=y_percep.mean()
            perceptr_var=y_percep.var()
            # Appending
            features.append(perceptr_mean)
            features.append(perceptr_var)

            # TEMPO
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            # Appending
            features.append(tempo)

            # MFCC (Mel-Frequency Cepstral Coefficients) 
            # (MEAN AND VARIANCE OF 20 MFCCS)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            # Finding mean and variance
            for i in range(len(mfccs)):
                mfccs_mean=mfccs[i].mean()
                mfccs_var=mfccs[i].var()
                # Appending
                features.append(mfccs_mean)
                features.append(mfccs_var)

            # Converting Feature extracted list into dataframe
            # ------------------------------------------------------------------------------>>>
            audio=pd.DataFrame(features).T
            # Normalising feature extracted audio data
            norm_audio= (audio - min_value)/(max_value - min_value)
            norm_audio= norm_audio.astype(float)

            # Training model and changing the values to numbers
            # ------------------------------------------------------------->>>
            model=Data.copy()

            model=model.replace('blues',0)
            model=model.replace('classical',1)
            model=model.replace('country',2)
            model=model.replace('disco',3)
            model=model.replace('hiphop',4)
            model=model.replace('jazz',5)
            model=model.replace('metal',6)
            model=model.replace('pop',7)
            model=model.replace('reggae',8)
            model=model.replace('rock',9)

            train_x = model.iloc[:, :-1].values
            train_y = model.iloc[:, 57].values

            # Applying XGBoost as it had highest accuracy
            # ------------------------------------------------------>>>
            preds = loaded_model.predict(norm_audio)

            # Creating Dictionary to decode the name of the genre
            dict={0:'Blues',1:'Classical',2:'Country',3:'Disco',4:'Hiphop',5:'Jazz',6:'Metal',7:'Pop',8:'Reggae',9:'Rock'}
            answer=dict[int(preds)]

    return render_template('output.html', prediction_text = answer)

app.run(debug=True)