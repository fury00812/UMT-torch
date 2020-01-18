import argparse
import datetime
import pathlib
import pickle
import torch
import numpy as np

from src.logger import create_logger
from src.data.loader import load_mono_data, check_all_data_params
from src.model.build_model import build_model

def get_parser():
    parser = argparse.ArgumentParser(description="UNMT training script")
    # experimental settings 
    parser.add_argument("-exp_name", type=str, required=True,
                        help="Experiment name")
    now = datetime.datetime.now()
    parser.add_argument("--exp_id", type=str, default="{0:%y%m%d-%H%M%S}".format(now),
                        help="Experiment ID")
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random generator seed (-1 for random)")
    # dataset
    parser.add_argument("--langs", type=str, default="",
                        help="Languages (lang1,lang2)")
    parser.add_argument("--mono_dataset", type=str, default="",
                        help="Monolingual dataset (lang1:trn,val,test;lang2:trn,val,test)")
    parser.add_argument("--max_len", type=int, default=10000,
                        help="Maximum length of sentences (after BPE)")
    # training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    return parser


def initialize_exp(params, exp_path):
    """Initialize the experience:
    :param params: command line argments
    :param exp_path: experiment path
    """
    # dump parameters
    params.dump_path = exp_path
    pickle.dump(params, open(params.dump_path/'params.pkl', 'wb'))

    # random seed
    if params.seed >= 0:
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)

    # create a logger
    logger = create_logger(params.dump_path/'train.log')
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)
    return logger



def main(params):
    # create experiment directories
    dump_path = pathlib.Path(params.dump_path)
    exp_path = pathlib.Path(dump_path/params.exp_name/params.exp_id)
    exp_path.mkdir(parents=True)

    # check parameters
    check_all_data_params(params)

    # initialize experiment
    logger = initialize_exp(params, exp_path)

    # load data
    data = load_mono_data(params)

    # build model
    #encoder, decoder, discriminator, lm = build_model(params, data) 
    encoder, decoder, lm = build_model(params, data)

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)
