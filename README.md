# README.txt

## Code Design

The code is structured as two pipelines of scripts. The following diagrams capture the dependency structure of the scripts (the following script depends on the output of the previous script):

![](./img/small-scale-pipeline.png?raw=True)
![](./img/large-scale-pipeline.png?raw=True)

## Modules

Install the listed dependencies for each of these modules -- following the instructions on each of their pages. 

### evaluate by HuggingFace

Link: [https://github.com/huggingface/evaluate](https://github.com/huggingface/evaluate)
Library Version: 0.4.0
Python Version: 3.8

### torcheval by PyTorch

Link: [https://github.com/pytorch/torcheval](https://github.com/pytorch/torcheval)
Library Version: 0.0.7
Python Version: 3.8

### WhisperX

Link: [https://github.com/m-bain/whisperX](https://github.com/m-bain/whisperX)
Version: 3.1.1

### OpenAI API
Link: [https://platform.openai.com/docs/introduction](https://platform.openai.com/docs/introduction)
Version: 1.9.0

### english-fisher-annotations

Link: [https://github.com/pariajm/english-fisher-annotations](https://github.com/pariajm/english-fisher-annotations)

We had to modify this code, so we provide the code here as a subdirectory.


### Spotify Podcast Dataset

Link: [https://podcastsdataset.byspotify.com/](https://podcastsdataset.byspotify.com/)

This dataset is maintained by Spotify, and access to the dataset is determined by Spotify.


### Additional Dependencies
* Pandas (Link: [https://pandas.pydata.org/](https://pandas.pydata.org/))
* tqdm (Link: [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm))
