#!/usr/bin/env python3

import argparse
import os

import fisher_annotator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model", type=str, default="./model/swbd_fisher_bert_Edev.0.9078.pt")
    parser.add_argument("--disfluency", type=bool, default=True)
    args = parser.parse_args()
    
    # print(args)

    labels = fisher_annotator.Annotate(
        input_path=args.input_path,
        output_path=args.output_path,
        model=args.model,
        disfluency=args.disfluency,
        )
    
    labels.setup()

if __name__ == "__main__":
    main()
