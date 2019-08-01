import os.path as osp
import argparse
import io
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=float, nargs=3, default=[1., 0., 0.])
parser.add_argument("--image_dir", type=str)
parser.add_argument("coco_json", type=argparse.FileType("r"))
args = parser.parse_args()

assert np.sum(args.split) == 1


def main():
    infile: io.TextIOWrapper = args.coco_json
    dataset:dict = json.load(infile)
    if args.image_dir:
        print(f"Using {args.image_dir} as the base path of all image files.")
        for img in dataset["images"]:
            fn = img["file_name"]
            base_fn = osp.basename(fn)
            img["file_name"] = osp.join(args.image_dir, base_fn)

    anns: list = dataset["annotations"]
    anns_orig = anns.copy()

    num_anns = len(anns)
    print(f"Number of annotations = {num_anns}")

    print(f"Splitting data: Training = {args.split[0]}, Validate = {args.split[1]}, Test = {args.split[2]}")
    num_training = int(np.round(num_anns * args.split[0]))
    num_validation = int(np.round(num_anns * args.split[1]))
    num_test = int(np.round(num_anns * args.split[2]))
    print(f"Annotations: Training = {num_training}, Validation = {num_validation}, Test = {num_test}")
    np.random.shuffle(anns)
    training, validation, testing = anns[:num_training], anns[num_training: num_validation], anns[-num_test:]



    print(f"Writing to coco.full.json")
    with open("coco.full.json", 'w') as f:
        json.dump(dataset, f)

    if num_training > 0:
        print(f"Writing to coco.training.json")
        with open("coco.training.json", 'w') as f:
            ds = dataset.copy()
            ds["annotations"] = training
            json.dump(ds, f)
    if num_validation > 0:
        print(f"Writing to coco.validation.json")
        with open("coco.validation.json", 'w') as f:
            ds = dataset.copy()
            ds["annotations"] = validation
            json.dump(ds, f)
    if num_test > 0:
        print(f"Writing to coco.test.json")
        with open("coco.test.json", 'w') as f:
            ds = dataset.copy()
            ds["annotations"] = testing
            json.dump(ds, f)

if __name__ == '__main__':
    main()
