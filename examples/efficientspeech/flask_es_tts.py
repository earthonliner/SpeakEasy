import torch
import yaml
import time
import numpy as np
import validators
import sounddevice as sd

from model import EfficientSpeech
from utils.tools import get_args, write_to_file
from synthesize import get_lexicon_and_g2p, text2phoneme
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def tts(lexicon, g2p, preprocess_config, model, text, args):
    text = text.replace('-', ' ')
    phoneme = np.array([text2phoneme(lexicon, g2p, text, preprocess_config, verbose=False)], dtype=np.int32)
    start_time = time.time()
    with torch.no_grad():
        phoneme = torch.from_numpy(phoneme).int().to(args.infer_device)
        wavs, lengths, _ = model({"phoneme": phoneme})
        wavs = wavs.cpu().numpy()
        lengths = lengths.cpu().numpy()
    elapsed_time = time.time() - start_time
    wav = np.reshape(wavs, (-1, 1))
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    # print(sampling_rate)
    wav_len = wav.shape[0] / sampling_rate
    real_time_factor = wav_len / elapsed_time
    if not args.play:
        write_to_file(wavs, preprocess_config, lengths=lengths, \
            wav_path=args.wav_path, filename=args.wav_filename)
    
    return wav, phoneme, wav_len, real_time_factor


if __name__ == '__main__':
    args = get_args()
    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    lexicon, g2p = get_lexicon_and_g2p(preprocess_config)
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    checkpoint = args.checkpoint
    model = EfficientSpeech.load_from_checkpoint(checkpoint, infer_device=args.infer_device, map_location=torch.device('cpu'))
    model = model.to(args.infer_device)
    model.eval()
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", backend="inductor")
    if args.play:
        sd.default.reset()
        sd.default.samplerate = sampling_rate
        sd.default.channels = 1
        sd.default.dtype = 'int16'
        sd.default.device = None
        sd.default.latency = 'low'
    def es_tts(text):
        wav = tts(lexicon, g2p, preprocess_config, model, text, args)[0]
        return wav


    from flask import Flask, request, jsonify
    import numpy as np
    app = Flask(__name__)
    @app.route('/convert_text_to_speech', methods=['POST'])
    def convert_text_to_speech():
        data = request.get_json()
        text = data.get('text')
        if text:
            wav_array = es_tts(text)
            return jsonify({'audio': wav_array.tolist()})
        else:
            return jsonify({'error': 'Text parameter is missing'}), 400

    app.run(debug=True)
