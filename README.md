# VNASR
Automatic Speech Recognition for Vietnamese

## Install dependencies
```
pip install -r requirements.txt
```

## [*Download*](https://commonvoice.mozilla.org/vi/datasets) the Commonvoice dataset

## Load dataset (for training only)
Convert the mp3 files to wav and export json data with the form `{key: "/path/to/audio", text: "transcript sentence"}`

Use:
```
py ./scripts/commonvoice_to_json.py --tsv_path /path/to/tsv/file --save_path /path/to/json/file
```

## Run server
```
cd ./server
py ./app.py
```
Server is then ready on port 6969

Request:
method = "POST"
enctype = "multipart/form-data"
key: audio (file type)      
value: upload the file you want to predict
