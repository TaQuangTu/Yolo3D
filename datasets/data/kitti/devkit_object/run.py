import argparse
import sys
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', type=str, default='max-min-area')
    parser.add_argument('--args', type=str, nargs='+', default=[''])
    opt = parser.parse_args()
    if opt.cmd == 'max-min-area':
        max_a, min_a = utils.calc_bbox_max_and_min_area(*opt.args)
        print('max min areas: %s, %s' % (max_a, min_a))
    elif opt.cmd == 'offset-vertex-to-center':
        utils.calc_offset_vertex_to_center(*opt.args, objs=['Car', 'Pedestrian', 'Cyclist'])
        # print('max min areas: %s, %s' % (max_a, min_a))
