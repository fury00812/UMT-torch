import os
import torch
from logging import getLogger
from .dataset import MonolingualDataset
from .dictionary import EOS_WORD, PAD_WORD, UNK_WORD, SPECIAL_WORD, SPECIAL_WORDS


logger = getLogger()

loaded_data = {}  # store binarized datasets in memory in case of multiple reloadings


def load_binarized(path, params):
    """
    Load a binarized dataset and log main statistics.
    :param path: *.pth
    - data['sentences']: 1d array with each sentence separated by '-1'
    - data['positions']: Positions of each sentence in data['sentences']
    - data['dico']: vocab dictionary {'hello': 105, 'world': 106, ...}
    - data['unk_words']: unknown dictionary {'popape': 3, 'nzzzza': 3, ...}
    """
    if path in loaded_data:
        logger.info("Reloading data loaded from %s ..." % path)
        return loaded_data[path]
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    loaded_data[path] = data
    data['positions'] = data['positions'].numpy()
    return data


def set_parameters(params, dico):
    """
    Define parameters / check dictionaries.
    """
    
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    blank_index = dico.index(SPECIAL_WORD % 0)
    bos_index = [dico.index(SPECIAL_WORD % (i + 1)) for i in range(params.n_langs)]
    if hasattr(params, 'eos_index'):
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.blank_index == blank_index
        assert params.bos_index == bos_index
    else:
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.blank_index = blank_index
        params.bos_index = bos_index


def check_all_data_params(params):
    """
    Check datasets parameters.
    """
    # check languages
    params.langs = params.langs.split(',')
    assert len(params.langs) == len(set(params.langs)) >= 2
    assert sorted(params.langs) == params.langs
    params.id2lang = {k: v for k, v in enumerate(sorted(params.langs))}
    params.lang2id = {k: v for v, k in params.id2lang.items()}
    params.n_langs = len(params.langs)


def load_mono_data(params):
    """
    Load monolingual data.
    """
    data = {'dico': {}, 'mono': {}}

    assert len(params.mono_dataset) != 0

    # str to dict{}
    params.mono_dataset = {k: v for k, v in [x.split(':') for x in params.mono_dataset.split(';') if len(x) > 0]}
    if len(params.mono_dataset) > 0:
        assert type(params.mono_dataset) is dict
        assert all(lang in params.langs for lang in params.mono_dataset.keys())
        assert all(len(v.split(',')) == 3 for v in params.mono_dataset.values())
        # convert values of dict{} to tuple() 
        params.mono_dataset = {k: tuple(v.split(',')) for k, v in params.mono_dataset.items()}
        assert all(all(((i > 0 and path == '') or os.path.isfile(path)) for i, path in enumerate(paths))
                   for paths in params.mono_dataset.values())

    for lang, paths in params.mono_dataset.items():
        assert lang in params.langs
        logger.info('============ Monolingual data (%s)' % lang)
        datasets = []

        for name, path in zip(['train', 'valid', 'test'], paths):
            if path == '':
                assert name != 'train'
                datasets.append((name, None))
                continue

            # load data
            mono_data = load_binarized(path, params)
            set_parameters(params, mono_data['dico'])
            # set / check dictionary
            if lang not in data['dico']:
                data['dico'][lang] = mono_data['dico']
            else:
                assert data['dico'][lang] == mono_data['dico']

            # monolingual data
            mono_data = MonolingualDataset(mono_data['sentences'], mono_data['positions'],
                                            data['dico'][lang], params.lang2id[lang], params)

            # remove too long sentences (train / valid only, test must remain unchanged)
            if name != 'test':
                mono_data.remove_long_sentences(params.max_len)
            # select a subset of sentences

        data['mono'][lang] = {k: v for k, v in datasets}

    logger.info('')
