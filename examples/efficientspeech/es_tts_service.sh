# base_eng_4M.ckpt / small_eng_952k.ckpt
python3 flask_es_tts.py \
  --checkpoint base_eng_4M.ckpt \
  --infer-device cpu \
  --n-blocks 3 \
  --reduction 2
