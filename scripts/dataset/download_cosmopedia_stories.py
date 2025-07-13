import argparse
import os
import os.path as osp
from datasets import load_dataset


def main(args):
    dataset_id = "khairi/cosmopedia-stories-young-children"
    save_dir = args.save_dir
    
    dataset = load_dataset(dataset_id)
    
    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    print(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save-dir', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
