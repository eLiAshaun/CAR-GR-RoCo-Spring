import os
import sys
from model.waft_a1 import ViTWarpV8
from model.waft_a2 import WAFTv2
from model.waft_rx import WAFTRX

def fetch_model(args):
    if args.algorithm == 'waft-a1':
        model = ViTWarpV8(args)
    elif args.algorithm == 'waft-a2':
        model = WAFTv2(args)
    elif args.algorithm == 'waft-rx':
        model = WAFTRX(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model
