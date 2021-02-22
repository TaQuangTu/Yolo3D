
import os
import random
import argparse


def split_datasets(dir,out_dir, splits, rates):
    splits = splits.split()
    rates = rates.split()
    file = os.listdir(dir)
    total_num = len(file)
    file = [file[i].split('.')[0] for i in range(total_num)]
    split_elem_num = [int(float(r)*total_num) for r in rates]
    split_num = len(splits)
    res = []
    for i in range(split_num - 1):
        # idx = [random.randint(0, total_num - 1) for _ in range(split_elem_num[i])]
        idx = random.sample(range(0, total_num), split_elem_num[i])
        sp = [file[j] for j in idx]
        file = [file[j] for j in range(0, total_num) if j not in idx]
        total_num = len(file)
        res.append(sp)
        pass
    res.append(file)
    for data, split in zip(res, splits):
        with open(os.path.join(out_dir, split+'.txt'), 'w') as f:
            f.writelines([d + '\n' for d in data])

def arg_parser():
    parser = argparse.ArgumentParser(description="Split Dataset")
    parser.add_argument("--data-path", default="", help="specific the dataset path")
    parser.add_argument("--out-path", default="", help="specific the output path")
    parser.add_argument("--splits", default='train test', help="specific splits eg. 'train, test' ")
    parser.add_argument("--scales", default='0.8 0.2', help="specific scales eg. '0.8, 0.2' ")
    return parser


if __name__ == '__main__':
    args = arg_parser().parse_args()
    split_datasets(args.data_path,
                   args.out_path,
                   args.splits,
                   args.scales)