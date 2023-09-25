#!/usr/bin/env sh
set -x # output commands to console

MODEL_NAME=${1:-stable-platypus2-13b.Q4_K_M.gguf}
DOWNLOAD_URL="https://huggingface.co/TheBloke/Stable-Platypus2-13B-GGUF/resolve/main/$MODEL_NAME"
SHA256SUM=84a425ee16b5e3edaddbb1e0969b8f37e5341b7a85ae53cd24dd7c76cbbd24cd
mkdir -p models

if [ -e "models/$MODEL_NAME" ]; then
    echo "SKIP: models/$MODEL_NAME already exists"
else
    wget $DOWNLOAD_URL -P models/
fi

actual_shasum=$(sha256sum models/$MODEL_NAME)
if [ $actual_shasum != $SHA256SUM ]; then
    echo "Unexpected sha256sum: $actual_shasum != $SHA256SUM"
    exit 1
fi

cd models && ln -f -s $MODEL_NAME model.gguf && cd ..
exit 0
