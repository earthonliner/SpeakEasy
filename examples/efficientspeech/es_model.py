import torch
import yaml
import time
import numpy as np
import validators

import sys
sys.path.append('./efficientspeech')
from model import EfficientSpeech
from utils.tools import write_to_file
from synthesize import get_lexicon_and_g2p, text2phoneme
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sounddevice as sd
sd.default.reset()
sd.default.samplerate = 22050
sd.default.channels = 1
sd.default.dtype = 'int16'
sd.default.device = None
sd.default.latency = 'low'


def tts(lexicon, g2p, preprocess_config, model, text, wav_path=None, wav_filename=None):
    text = text.strip()
    text = text.replace('-', ' ')
    phoneme = np.array(
            [text2phoneme(lexicon, g2p, text, preprocess_config, verbose=False)], dtype=np.int32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        phoneme = torch.from_numpy(phoneme).int().to(device)
        wavs, lengths, _ = model({"phoneme": phoneme})
        wavs = wavs.cpu().numpy()
        lengths = lengths.cpu().numpy()
    wav = np.reshape(wavs, (-1, 1))
    write_to_file(wavs, preprocess_config, lengths=lengths, \
        wav_path=wav_path, filename=wav_filename)
    
    return wav

def get_es_model():
    preprocess_config_path = './efficientspeech/config/LJSpeech/preprocess.yaml'
    checkpoint_path = './efficientspeech/small_eng_952k.ckpt'
    hifigan_checkpoint_path = './efficientspeech/hifigan/LJ_V2/generator_v2'
    preprocess_config = yaml.load(open(preprocess_config_path), Loader=yaml.FullLoader)
    preprocess_config["path"]["lexicon_path"] = './efficientspeech/lexicon/librispeech-lexicon.txt'
    preprocess_config["path"]["preprocessed_path"] = './efficientspeech/preprocessed_data/LJSpeech'
    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientSpeech(preprocess_config=preprocess_config, 
                            infer_device=device,
                            hifigan_checkpoint=hifigan_checkpoint_path,)
    model = model.load_from_checkpoint(checkpoint_path,
                                       infer_device=device,
                                       map_location=torch.device('cpu'))
    model = model.to(device)
    model.eval()
    
    return lexicon, g2p, preprocess_config, model
    # default number of threads is 128 on AMD
    # this is too high and causes the model to run slower
    # set it to a lower number eg --threads 24 
    # if args.threads is not None:
    #     torch.set_num_threads(args.threads)

