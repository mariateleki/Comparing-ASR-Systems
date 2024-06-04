"""
Pre-processing and annotating Fisher transcripts using 
a SOTA joint parser and disfluency detector model. For 
a complete description of the model, please refer to 
the following paper:
https://www.aclweb.org/anthology/2020.acl-main.346.pdf


* DisfluencyTagger --> finds disfluency labels
* Parser --> finds constituency parse trees
* Annotate --> pre-processes transcripts for annotation

(c) Paria Jamshid Lou, 14th July 2020.
"""

import codecs
import fnmatch
import os
import re   
import torch

import parse_nk

import shutil

import re
import utils_trees

# allows the import of utils files from the upper directory
import sys
sys.path.append("..")

import utils_general

import traceback
import logging


class DisfluencyTagger:
    """
    This class is called when self.disfluency==True.    

    Returns:
        A transcript with disfluency labels:
            e.g. "she E she _ likes _ movies _"
            where "E" indicate that the previous 
            word is disfluent and "_" shows that 
            the previous word is fluent.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
 
    @staticmethod
    def fluent(tokens):
        leaves_tags = [t.replace(")","")+" _" for t in tokens if ")" in t]      
        return " ".join(leaves_tags)

    @staticmethod
    def disfluent(tokens):
        # remove first and last brackets
        tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
        open_bracket, close_bracket, pointer = 0, 0, 0      
        df_region = False
        tags = []
        while pointer < len(tokens):
            open_bracket += tokens[pointer].count("(")                
            close_bracket += tokens[pointer].count(")")
            if "(EDITED" in tokens[pointer]:  
                open_bracket, close_bracket = 1, 0             
                df_region = True
            elif ")" in tokens[pointer]:
                label = "E" if df_region else "_"  
                tags.append(
                    (tokens[pointer].replace(")", ""), label)
                    )                 
            if all(
                (close_bracket,
                open_bracket == close_bracket)
                ):
                open_bracket, close_bracket = 0, 0
                df_region = False            

            pointer += 1
        return " ".join(list(map(lambda t: " ".join(t), tags)))


class Parser(DisfluencyTagger):
    """
    Loads the pre-trained parser model to find silver parse trees     
   
    Returns:
        Parsed and disfluency labelled transcripts
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def torch_load(self):
        if parse_nk.use_cuda:
            return torch.load(
                self.model
                )
        else:
            return torch.load(
                self.model, 
                map_location=lambda storage, 
                location: storage,
                )

    def run_parser(self, input_sentences):
        eval_batch_size = 1
        # print("Loading model from {}...".format(self.model))
        assert self.model.endswith(".pt"), "Only pytorch savefiles supported"

        info = self.torch_load()
        assert "hparams" in info["spec"], "Older savefiles not supported"
        parser = parse_nk.NKChartParser.from_spec(
            info["spec"], 
            info["state_dict"],
            )

        # print("Parsing sentences...")
        sentences = [sentence.split() for sentence in input_sentences]
        # Tags are not available when parsing from raw text, so use a dummy tag
        if "UNK" in parser.tag_vocab.indices:
            dummy_tag = "UNK"
        else:
            dummy_tag = parser.tag_vocab.value(0)
        
        all_predicted = []
        for start_index in range(0, len(sentences), eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+eval_batch_size]
            subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
            predicted, _ = parser.parse_batch(subbatch_sentences)
            del _
            all_predicted.extend([p.convert() for p in predicted])
        
        parse_trees, df_labels = [], []
        for tree in all_predicted:          
            linear_tree = tree.linearize()
            parse_trees.append(linear_tree)
            if self.disfluency:
                tokens = linear_tree.split()
                # disfluencies are dominated by EDITED nodes in parse trees
                if "EDITED" not in linear_tree: 
                    df_labels.append(self.fluent(tokens))
                else:
                    df_labels.append(self.disfluent(tokens))
                    
        return parse_trees, df_labels

           
class Annotate(Parser):   
    """
    Writes parsed and disfluency labelled transcripts into 
    *_parse.txt and *_dys.txt files, respectively.

    """ 
    def __init__(self, **kwargs):
        self.input_path = kwargs["input_path"]
        self.output_path = kwargs["output_path"] 
        self.model = kwargs["model"] 
        self.disfluency = kwargs["disfluency"] 

    def setup(self): 
        self.parse_sentences()

    def parse_sentences(self):
        
        # input
        input_filepath = self.input_path
        input_filename = os.path.basename(input_filepath)
        doc = self.read_transcription(input_filepath)
        
        # output
        output_dir = os.path.dirname(self.output_path)
        parse_filename = os.path.basename(self.output_path)
        
        # try:
        parse_trees, df_labels = self.run_parser(doc) # so this doc needs to be a list of sentences

        # Write constituency parse trees and disfluency labels into files
        new_filename = os.path.join(output_dir, parse_filename.replace(".txt", "_parse.txt"))
        utils_general.delete_file_if_already_exists(new_filename)
        with open(new_filename, "w") as output_file:
            output_file.write("\n".join(parse_trees))

        new_filename = os.path.join(output_dir, parse_filename.replace(".txt", "_orig_dys.txt"))
        utils_general.delete_file_if_already_exists(new_filename)
        with open(new_filename, "w") as output_file:
            output_file.write("\n".join(df_labels))

        new_text = utils_trees.get_intj_prn_edited_transcript_from_tree_file(filepath=os.path.join(output_dir, parse_filename.replace(".txt", "_parse.txt")))
        new_filename = os.path.join(output_dir, parse_filename.replace(".txt", "_dys.txt"))
        utils_general.delete_file_if_already_exists(new_filename)
        with open(new_filename, "w") as output_file:
            output_file.write(new_text)

        # except Exception as e:
        #     print("Exception:", e)
        #     traceback.print_exc()
                    
        return
    
    # original function split into sentences based on short swb conversations
    # we adapt this function to have it split into sentences based on period (full stop) locations
    def read_transcription(self, trans_file):  
        with open(trans_file) as f:
            contents = f.read()
            
        # split into sentences
        sentences = contents.replace("!",".").replace("?",".").split(".")  # split on sentences
        
        # limit sentences to 300 words for compatibility with the model architecture: https://github.com/nikitakit/self-attentive-parser/issues/37
        limited_sentences = []
        for sentence in sentences:
            limited_sentences.append(" ".join(sentence.split(" ")[0:300]))
        
        # clean the sentences for compatibility with the model
        cleaned_sentences = []
        for sentence in limited_sentences:

            # split into words within the sentences, clean
            tokens = sentence.split(" ")
            cleaned_tokens = []
            for token in tokens:
                token = self.validate_transcription(token)
                if token is not None:
                    cleaned_tokens.append(token)
            cleaned_sentence = " ".join(cleaned_tokens)

            if(len(cleaned_sentence) > 0):
                cleaned_sentences.append(cleaned_sentence)
                
        # print("cs:", cleaned_sentences)
                
        return cleaned_sentences

    @staticmethod
    def validate_transcription(label):
        if re.search(r"[0-9]|[(<\[\]&*{]", label):
            return None

        label = label.replace("_", " ")
        label = re.sub("[ ]{2,}", " ", label)
        label = label.replace(".", "")
        label = label.replace(",", "")
        label = label.replace(";", "")
        label = label.replace("?", "")
        label = label.replace("!", "")
        label = label.replace(":", "")
        label = label.replace("\"", "")
        label = label.replace("'re", " 're")
        label = label.replace("'ve", " 've")
        label = label.replace("n't", " n't")
        label = label.replace("'ll", " 'll")
        label = label.replace("'d", " 'd")
        label = label.replace("'m", " 'm")
        label = label.replace("'s", " 's")
        label = label.strip()
        label = label.lower()

        return label if label else None   
