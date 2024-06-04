import os
import shutil

TOTAL_NUM_TEST_FILES = 1027
TOTAL_NUM_TRAIN_FILES = 105360

PATH_TO_PROJECT = "/home/grads/m/mariateleki/analysis-asr"     
PATH_TO_SPOTIFY_NO_AUDIO = "/data2/maria/Spotify-Podcasts/podcasts-no-audio-13GB" 

# project dirs
PATH_TO_FIGS = os.path.join(PATH_TO_PROJECT, "figs")
PATH_TO_CSV = os.path.join(PATH_TO_PROJECT, "csv")

# json
PATH_TO_SPOTIFY_TEST = os.path.join(PATH_TO_SPOTIFY_NO_AUDIO, "TREC/spotify-podcasts-2020")
PATH_TO_2020_TEST_DIR = os.path.join(PATH_TO_SPOTIFY_TEST, "podcasts-transcripts-summarization-testset")
PATH_TO_2020_TEST_DF = os.path.join(PATH_TO_SPOTIFY_TEST, "metadata-summarization-testset.tsv")
PATH_TO_TRAIN_DIR = os.path.join(PATH_TO_SPOTIFY_NO_AUDIO, "spotify-podcasts-2020/podcasts-transcripts")
PATH_TO_TRAIN_DF = os.path.join(PATH_TO_SPOTIFY_NO_AUDIO, "metadata.tsv")

# audio/ogg
PATH_TO_AUDIO_TRAIN_DIR = os.path.join("/data2/maria/Spotify-Podcasts/podcasts-audio-only-2TB/podcasts-audio")
PATH_TO_AUDIO_TEST_DIR = os.path.join("/data2/maria/Spotify-Podcasts/podcasts-audio-only-2TB/podcasts-audio-summarization-2020-testset")

# treebank-3 & annotation
PATH_TO_TREEBANK = os.path.join(PATH_TO_PROJECT, "treebank_3")
PATH_TO_ANNOTATION_DIR = "/hddc/annotation-dir/podcasts-no-audio-13GB/spotify-podcasts-2020/podcasts-transcripts"

def write_file(new_filename, directory, text):
    new_filepath = os.path.join(directory, new_filename)
    with open(new_filepath, mode="w") as f:
        f.write(text)
        
def write_file_if_not_blank(file, text):
    if text:
        with open(file, mode="w") as f:
            f.write(text)
            
def read_file(filepath):
    text = ""
    with open(filepath) as f:
        text = f.read()
    return text

def delete_file_if_already_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def create_and_or_clear_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)
    # if this dir does exist, clear it out and re-create it
    else:
        shutil.rmtree(d)
        os.mkdir(d)
        
def just_create_this_dir(d):
    # check if all the necessary dirs exist, or need to be created
    if not os.path.isdir(d):
        os.mkdir(d)
        
def copy_all_files(input_dir, output_dir):
    
    print("input_dir:", input_dir, "\noutput_dir:", output_dir, "\n")
    print("# of files in input_dir:", len(os.listdir(input_dir)))

    for file in os.listdir(input_dir):
        path_to_file = os.path.join(input_dir, file)

        text = ""
        text = utils_general.read_file(path_to_file)

        utils_general.write_file(new_filename=file, directory=output_dir, text=text)
        
    print("# of files in output_dir after copy:", len(os.listdir(output_dir)), "\n\n")
    
def count_num_files(input_dir, file_extension):
    num_files = 0
    for root, dirs, files in os.walk(input_dir):
        if files:
            files_matching_extension = [f for f in files if f.endswith(file_extension)]
            num_files += len(files_matching_extension)
    return num_files
    
    