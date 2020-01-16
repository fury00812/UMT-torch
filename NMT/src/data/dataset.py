from logging import getLogger
import math
import numpy as np
import torch


logger = getLogger()


class Dataset(object):

    def __init__(self, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.unk_index = params.unk_index
        self.bos_index = params.bos_index
        self.batch_size = params.batch_size

    def batch_sentences(self, sentences, lang_id):
        """
        """
        pass


class MonolingualDataset(Dataset):

    def __init__(self, sent, pos, dico, lang_id, params):
        super(MonolingualDataset, self).__init__(params)
        assert type(lang_id) is int
        self.sent = sent
        self.pos = pos
        self.dico = dico
        self.lang_id = lang_id
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.is_parallel = False

        # check number of sentences
        assert len(self.pos) == (self.sent == -1).sum()

        self.remove_empty_sentences()

        # check sentences indices
        assert len(pos) == (sent[torch.from_numpy(pos[:, 1])] == -1).sum()
        # check dictionary indices
        assert -1 <= sent.min() < sent.max() < len(dico)
        # check empty sentences
        assert self.lengths.min() > 0

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))
