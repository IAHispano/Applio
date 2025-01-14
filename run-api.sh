#!/bin/sh
printf "\033]0;Applio\007"
. .venv/bin/activate

 export PYTORCH_ENABLE_MPS_FALLBACK=1
 export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
 
uvicorn  api:app  --host 0.0.0.0 --port 9033
