import json
import argparse

parser = argparse.ArgumentParser(
    description="Returns the context, question, and answer.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--truth_filename",
                    default="data-release/BioASQ-6b/test/Full-Abstract/BioASQ-test-factoid-6b-3.json",
                    help="the path to the ground truth file")

parser.add_argument("--predictions_filename",
                    default="/tmp/factoid_output/predictions.json",
                    help="the path to the predictions file")
args = parser.parse_args()

with open(args.truth_filename) as json_file2:
    truths = json.load(json_file2)

with open(args.predictions_filename) as json_file:
    answers = json.load(json_file)

for data in answers.items():

    id = data[0]
    answer = data[1]
    print("ID: {}".format(id))
    print("="*32)

    for d in truths["data"][0]["paragraphs"]:
        if d["qas"][0]["id"] == id:
            question = d["qas"][0]["question"]
            context = d["context"]
            print("Context:\n********\n{}\n\nQuestion:\n*********\n{}\n\nPrediction:\n***********\n{}".format(context, question, answer))

    print("\n\n")
