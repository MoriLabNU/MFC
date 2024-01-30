import argparse
import json
import os.path as osp

import mmcv
import numpy as np
# from cityscapesscripts.preparation.json2labelImg import json2labelImg
from PIL import Image


def convert_json_to_label(json_file):
    label_file = json_file

    pil_label = Image.open(label_file)
    label = np.asarray(pil_label)[:, :, 0]
    id_to_trainid = {
        50: 0,
        11: 1,
        21: 2,
        31: 3,
        32: 3,
        12: 4,
        22: 5,
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = label_file.replace('.png', '_labelTrainIds.png')
    assert label_file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)

    if 'train/' in json_file:
        return sample_class_stats
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert CS8K annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gt', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(gt_dir, '_mask.png', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)
    print(len(poly_files))

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
