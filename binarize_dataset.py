from preprocessing import DataProvider
import torch
import argparse
from config import config


def main(args):
    dataloader = DataProvider(config, True, "train", args.make_train, args.make_valid, args.provide_valid)
    if args.make_train:
        train_dataset = dataloader.dataset
        torch.save(train_dataset, args.train_path + ".pt")
    if args.make_valid:
        validation_dataset = dataloader.vali_dataset
        torch.save(validation_dataset, args.valid_path + ".pt")
    print("Preprocessing finished!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--make_train", action="store_true")
    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--valid_path", type=str, default="")
    parser.add_argument("--make_valid", action="store_true")
    parser.add_argument("--provide_valid", action="store_true")
    args = parser.parse_args()    
    main(args)

