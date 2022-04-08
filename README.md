# VNASR
Automatic Speech Recognition for Vietnamese

## Install modules
```
pip3 install -r requirements.txt
```

## [*Download*](https://commonvoice.mozilla.org/vi/datasets) the Commonvoice dataset

## Load dataset
Convert the mp3 files to wav and export json data with the form `{key: "/path/to/audio", text: "transcript sentence"}`
Use
```
python3 ./scripts/commonvoice_to_json.py --tsv_path /path/to/tsv/file --save_path /path/to/json/file
```