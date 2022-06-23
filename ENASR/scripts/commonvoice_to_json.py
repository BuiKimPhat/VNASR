"""
Script to convert commonvoice into .wav and create the training and test json files for speechrecognition.
"""
import os
import argparse
import json
import random
import csv
from pydub import AudioSegment

def main(args):
    data = []
    directory = args.tsv_path.rpartition('/')[0]
    dest_dir = args.tsv_save
    percent = args.percent
    
    # count number of lines in the .tsv file
    with open(args.tsv_path) as f:
        length = sum(1 for line in f) - 1
    
    # read csv file
    with open(args.tsv_path, newline='') as csvfile: 
        reader = csv.DictReader(csvfile, delimiter=',')
        index = 1
        if(args.convert):
            print(str(length) + " files found")
        for row in reader:
            if index > 24000:
                break
            file_name = row['filename'] # with extension .mp3
            filename = file_name.rpartition('.')[0] + ".wav" # with new extension .wav
            text = row['text']
            if(args.convert):
                data.append({
                "key": dest_dir + "/cv-valid-train/" + filename,
                "text": text
                })
                print("converting file " + str(index) + "/" + str(length) + " to wav")
                src = directory + "/cv-valid-train/" + file_name
                dst = dest_dir + "/cv-valid-train/" + filename
                sound = AudioSegment.from_mp3(src)
                sound.export(dst, format="wav")
                index = index + 1
            else:
                data.append({
                "key": directory + "/cv-valid-train/" + filename,
                "text": text
                })
                
    random.shuffle(data)
    print("Creating JSON files...")

    # create train.json
    f = open(args.save_path +"/"+ "train.json", "w")
    with open(args.save_path +"/"+ 'train.json','w', encoding='utf8') as f:
        d = len(data)
        i=0
        while(i<int(d-d*percent/100)):
            r=data[i]
            line = json.dumps(r, ensure_ascii=False)
            f.write(line + "\n")
            i = i+1
    
    # create test.json
    f = open(args.save_path +"/"+ "test.json", "w")
    with open(args.save_path +"/"+ 'test.json','w', encoding='utf8') as f:
        d = len(data)
        i=int(d-d*percent/100)
        while(i<d):
            r=data[i]
            line = json.dumps(r, ensure_ascii=False)
            f.write(line + "\n")
            i = i+1
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Script to convert commonvoice into .wav and create the training and test json files for speechrecognition.
    """)
    parser.add_argument('--tsv_path', type=str, default=None, required=True,
                        help='path to one of the .tsv files found in cv-corpus')
    parser.add_argument('--tsv_save', type=str, default=None, required=True,
                        help='path to save wav files')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to the dir where the json files are supposed to be saved')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='percent of clips put into test.json instead of train.json')
    parser.add_argument('--convert', default=True, action='store_true',
                        help='says that the script should convert mp3 to wav')
    parser.add_argument('--not-convert', dest='convert', action='store_false',
                        help='says that the script should not convert mp3 to wav')
    args = parser.parse_args()
    
    main(args)