import os
import logging
import torch.distributed as dist

class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank

    def filter(self, record):
        return dist.get_rank() == self.rank

def create_logger(log_path):
    if os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)  
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger