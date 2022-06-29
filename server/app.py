from unicodedata import name
from flask import Flask, request
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from model import SpeechRecognition
from utils import DecodeGreedy
import os

SAVE_PATH = 'trained_model.pth' # model save path

app = Flask(__name__)

h_params = SpeechRecognition.hyper_parameters
model = SpeechRecognition(**h_params)
model.load_state_dict(torch.load(SAVE_PATH, map_location=torch.device('cpu'))["model_state_dict"])
model.eval()

featurizer = torchaudio.transforms.MelSpectrogram(sample_rate=8000, n_mels=80)

def predict(filename):
    with torch.no_grad():
        waveform, sr = torchaudio.load(filename)
        inputs = np.log(featurizer(waveform) + 1e-20) # (channels, features, time)
        hidden = (torch.zeros(1, 1, 1024),
                torch.zeros(1, 1, 1024))
        model_in = inputs.unsqueeze(1) # (channels, batch_size = 1, features, time)
        out, _ = model(model_in, hidden)  # (time, batch, n_class)
        out = torch.nn.functional.softmax(out, dim=2) 
        out = out.transpose(0, 1) # (batch, time, n_class)
        # out_args = None
        # out_args = out if out_args is None else torch.cat((out_args, out), dim=1)
        return DecodeGreedy(out)
	
@app.route('/', methods = ['GET','POST'])
def predict_audio():
    save_name = "audio_temp.wav"
    if request.method == 'POST':
        f = request.files['audio']
        if f.filename.split(".")[-1] == "wav":
            f.filename = save_name
            f.save(f.filename)
            return "Speech to text: " + predict(save_name)
        else:
            return "Error: File must be in .wav format"
    else: 
        return """
            <form action="/" method="post" enctype = "multipart/form-data">
                <input name="audio" type="file" />
                <input type="submit" />
            </form>
        """
        
app.run(host="0.0.0.0", port=6969)