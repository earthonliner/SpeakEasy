# change the model path of yours
CHAT_MODEL=/Users/xuliu/Documents/vscode/llm/model_zoo/chatglm3-ggml-q4_0.bin
ASR_MAIN=/Users/xuliu/Documents/vscode/speech/whisper.cpp/main
ASR_MODEL=/Users/xuliu/Documents/vscode/speech/whisper.cpp/models/ggml-base.en.bin
TTS_MODEL=tts_models/en/ljspeech/vits--neon
python3 cli_demo.py \
  --model ${CHAT_MODEL} \
  --interactive \
  --runtime_dir './runtime' \
  --asr_main ${ASR_MAIN} \
  --asr_model ${ASR_MODEL} \
  --tts_model ${TTS_MODEL} \
  --input_device 1

