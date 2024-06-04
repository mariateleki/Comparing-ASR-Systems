import os
import re

import tb

# adapted from tb.py, yields leaf nodes
def get_annotated_transcript(tree):
    """Yields the terminal or leaf nodes of tree."""
    
    def disfluent_visit(node):
        if isinstance(node, list):
            for child in node[1:]:
                yield from disfluent_visit(child)
        else:
            yield node + " E"
    
    def visit(node):
        if isinstance(node, list):
            for child in node[1:]:
                if any(x in ["EDITED", "INTJ", "PRN"] for x in child):
                    yield from disfluent_visit(child)
                else:
                    yield from visit(child)
        else:
            yield node + " _"
    yield from visit(tree) 

# this function is used for getting (our way of) formatting transcripts from the trees
def get_intj_prn_edited_transcript_from_tree_file(filepath):
    tb_file = tb.read_file(filepath)
    
    line_annotations = []
    for line in tb_file:
        line_annotations.append(" ".join(list(get_annotated_transcript(line))))
        
    return " ".join(line_annotations)
