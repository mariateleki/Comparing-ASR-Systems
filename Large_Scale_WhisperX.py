"""
Large_Scale_WhisperX.py

This script:
* Iterates through the files by walking the directory.
* Truncates the audio files to 10 minutes using ffmpeg and writing to a temp location.
* Writes out the output json from WhisperX.

WhisperX: https://github.com/m-bain/whisperX
Warnings are ok to ignore: https://github.com/m-bain/whisperX/issues/258

CUDA_VISIBLE_DEVICES=0 python Large_Scale_WhisperX.py

Issue from Spotify, will see this being worked around in the code: 
" (aasishp@spotify.com)"
" (aasishp@spotify.com 2)"
"""

import os
import json
import pathlib
import subprocess
import logging
from datetime import datetime
import argparse
from tqdm import tqdm
import pandas as pd
import whisperx
import gc

# allows the import of utils files from the upper directory
import sys
sys.path.append("..")
import utils_general
import utils_podcasts

# set variables
module_name = "whisperx"
time_amount = "10min"
split_name = "train"

# set up logging
utils_general.just_create_this_dir("./logs")
logging.basicConfig(filename=f"./logs/{module_name}-{datetime.now().isoformat(timespec='seconds')}.log", encoding="utf-8", level=logging.DEBUG)

# initialize whisperx
device = "cuda" 
batch_size = 24
compute_type = "float16"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# helpful function
def get_subprocess_cmd(input_path, output_path, time_to_truncate_to_in_seconds):
    # remove "-loglevel error" to show traditional ffmpeg output
    cmd = f"""ffmpeg -hide_banner -loglevel error -ss 0 -t {time_to_truncate_to_in_seconds} -i"""
    cmd = cmd.split()
    
    # the paths may have spaces in them from the Spotify dataset, so their paths get appended next
    cmd.append(f"{input_path}")
    cmd.append(f"{output_path}")
    
    return cmd

# create dir for temp files created by ffmpeg
utils_general.just_create_this_dir("./temp-files")

# initialize the progress bar
pbar = tqdm(total=utils_general.TOTAL_NUM_TEST_FILES)

# truncate all the files for this combo
for root, dirs, files in os.walk(utils_general.PATH_TO_AUDIO_TEST_DIR):
    if files:

        # # create the output dir structure (in the same way as the input dir)
        out_root = f"/data1/maria/Spotify-Podcasts/{split_name}-{time_amount}-{module_name}-dir"
        pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)

        for file in files:
            
            # set up the (potential) output filepath for each file
            output_filepath = os.path.join(out_root, file.replace(".ogg",""), "transcript.json")
            pathlib.Path(os.path.dirname(output_filepath)).mkdir(parents=True, exist_ok=True)
            
            # if this script hasn't already created the transcription (ex: re-running due to CUDA out of memory errors)
            if not os.path.exists(output_filepath):

                # set up input filepath
                input_filepath = os.path.join(root, file)

                # set up temp result filepath for ffmpeg
                temp_result_filepath = f"./temp-files/temp-result-{module_name}-{split_name}.ogg"
                utils_general.delete_file_if_already_exists(temp_result_filepath)
                
                # trim and convert the file
                result = subprocess.run(get_subprocess_cmd(input_path=input_filepath, 
                                                  output_path=temp_result_filepath, 
                                                  time_to_truncate_to_in_seconds=10*60))  # 10 min * 60 seconds/min 

                # transcribe with whisperx
                print(input_filepath, output_filepath)
                audio = whisperx.load_audio(temp_result_filepath)
                result = model.transcribe(audio, batch_size=batch_size)
                    
                try: 

                    # write to file
                    with open(output_filepath, "w") as f:
                        json.dump(result, f)
                        
                        
                except Exception as e:
                    logging.debug(e)
                    

            # update the progress bar (because this file is completed)
            pbar.update(1)

# close the progress bar
pbar.close()
print()
