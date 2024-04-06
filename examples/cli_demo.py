import os
import argparse
from pathlib import Path
from typing import List
import requests
import numpy as np
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")
import chatglm_cpp
import sounddevice as sd

from chat_utils import remove_non_english_chars, run_terminal_command
from audio_utils import record_and_save_audio
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "chatglm-ggml.bin"
BANNER = """ ==================== Powered by ChatGLM.cpp ====================
    ________          __  ________    __  ___                 
   / ____/ /_  ____ _/ /_/ ____/ /   /  |/  /_________  ____  
  / /   / __ \/ __ `/ __/ / __/ /   / /|_/ // ___/ __ \/ __ \ 
 / /___/ / / / /_/ / /_/ /_/ / /___/ /  / // /__/ /_/ / /_/ / 
 \____/_/ /_/\__,_/\__/\____/_____/_/  /_(_)___/ .___/ .___/  
                                              /_/   /_/       
""".strip("\n")
WELCOME_MESSAGE = "Welcome to ChatGLM.CPP-based oral English bot! Ask whatever you want. Say 'clear' to clear context. Say 'stop' to exit."
tts_url = 'http://localhost:5000/convert_text_to_speech'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_PATH, type=str, help="model path")
    parser.add_argument("--mode", default="chat", type=str, choices=["chat", "generate"], help="inference mode")
    parser.add_argument("-p", "--prompt", default="Hi", type=str, help="prompt to start generation with")
    parser.add_argument("--pp", "--prompt_path", default=None, type=Path, help="path to the plain text file that stores the prompt")
    parser.add_argument("-s", "--system", default=None, type=str, help="system message to set the behavior of the assistant")
    parser.add_argument("--sp", "--system_path", default=None, type=Path, help="path to the plain text file that stores the system message")
    parser.add_argument("-i", "--interactive", action="store_true", help="run in interactive mode")
    parser.add_argument("-l", "--max_length", default=2048, type=int, help="max total length including prompt and output")
    parser.add_argument("--max_new_tokens", default=-1, type=int, help="max number of tokens to generate, ignoring the number of prompt tokens")
    parser.add_argument("-c", "--max_context_length", default=2048, type=int, help="max context length")
    parser.add_argument("--top_k", default=0, type=int, help="top-k sampling")
    parser.add_argument("--top_p", default=0.7, type=float, help="top-p sampling")
    parser.add_argument("--temp", default=0.95, type=float, help="temperature")
    parser.add_argument("--repeat_penalty", default=1.0, type=float, help="penalize repeat sequence of tokens")
    parser.add_argument("-t", "--threads", default=0, type=int, help="number of threads for inference")
    parser.add_argument("--runtime_dir", default=None, type=str, help="path to save chat audio and text file")
    parser.add_argument("--asr_main", default=None, type=str, help="path to run asr main program")
    parser.add_argument("--asr_model", default=None, type=str, help="path to load asr model")
    parser.add_argument("--input_device", default=0, type=int, help="the index of the input device")
    args = parser.parse_args()

    prompt = args.prompt
    if args.pp:
        prompt = args.pp.read_text()
    system = args.system
    if args.sp:
        system = args.sp.read_text()
    os.makedirs(args.runtime_dir, exist_ok=True)
    sampling_rate = 22050            
    sd.default.samplerate = sampling_rate
    sd.default.latency = 'low'
    pipeline = chatglm_cpp.Pipeline(args.model)
    if args.mode != "chat" and args.interactive:
        print("interactive demo is only supported for chat mode, falling back to non-interactive one")
        args.interactive = False
    generation_kwargs = dict(
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        do_sample=args.temp > 0,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temp,
        repetition_penalty=args.repeat_penalty,
        stream=True,
    )

    system_messages: List[chatglm_cpp.ChatMessage] = []
    if system is not None:
        system_messages.append(chatglm_cpp.ChatMessage(role="system", content=system))

    messages = system_messages.copy()

    if not args.interactive:
        if args.mode == "chat":
            messages.append(chatglm_cpp.ChatMessage(role="user", content=prompt))
            for chunk in pipeline.chat(messages, **generation_kwargs):
                print(chunk.content, sep="", end="", flush=True)
        else:
            for chunk in pipeline.generate(prompt, **generation_kwargs):
                print(chunk, sep="", end="", flush=True)
        print()
        return

    print(BANNER)
    print()
    print(WELCOME_MESSAGE)
    print()

    prompt_width = len(pipeline.model.config.model_type_name)

    if system:
        print(f"{'System':{prompt_width}} > {system}")

    talk_round = 0
    while True:
        try:
            init_prompt = ""
            if talk_round == 0:
                init_prompt = "Assuming that you are an English teacher and please help me practice my English, " + \
                                "and you should ask me questions to encourage me talk more."
            input_prompt = f"{'Prompt':{prompt_width}} > "
            role = "user"
            my_audio_path = os.path.join(args.runtime_dir, f"chat_{talk_round}_user.wav")
            my_text_path = os.path.join(args.runtime_dir, f"chat_{talk_round}_user.wav.txt")
            talk_round += 1
            record_and_save_audio(save_path=my_audio_path, channels=args.input_device)
            asr_cmd = f'{args.asr_main} -m {args.asr_model} -f {my_audio_path} -otxt {my_text_path}'
            run_terminal_command(asr_cmd)
            my_audio_prompt = ' '.join(open(my_text_path).readlines()).strip().replace('[BLANK_AUDIO]', '').strip()
            constrain_prompt = ' and please reply in English no more than two sentences.'
            prompt = input_prompt + my_audio_prompt + init_prompt + constrain_prompt
        except Exception as e:
            print('Encountered errors: ', e)
            break

        if not prompt:
            continue
        if prompt == "stop":
            break
        if prompt == "clear":
            messages = system_messages
            continue

        messages.append(chatglm_cpp.ChatMessage(role=role, content=prompt))
        print(f"{pipeline.model.config.model_type_name} > ", sep="", end="")
        chunks = []
        for chunk in pipeline.chat(messages, **generation_kwargs):
            print(chunk.content, sep="", end="", flush=True)
            chunks.append(chunk)
        response = ''.join([x.content for x in chunks]).strip()
        response = remove_non_english_chars(response)
        print()
        msg_out = pipeline.merge_streaming_messages(chunks)
        messages.append(msg_out)
        ai_audio_path = f'{args.runtime_dir}/chat_{talk_round}_ai.wav'  
        response_txt = remove_non_english_chars(msg_out.content.replace('\n', ' ').strip())
        if response_txt:
            response = requests.post(tts_url, json={'text': response_txt})
            if response.status_code == 200:
                wav = response.json()['audio']
                wav = np.array([x[0] for x in wav], dtype=np.float32).reshape((-1, 1))
                wavfile.write(ai_audio_path, sampling_rate, wav)
                sd.play(wav)
                sd.wait()
            else:
                print('Error:', response.json())

    print("Bye")


if __name__ == "__main__":
    main()
