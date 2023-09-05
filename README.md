# LLM Demo

A simple gradio app to demonstrate
how (current) language models (LLMs) operate.
This demo demystifies LLMs which generate text token-by-token
simply by feeding back the output token into the context.
In particular this demo is not intended to be used in production.


## Usage


### Install dependencies

`python -m venv .env && pip install -U pip && pip install -r requirements.txt`

### Download model bin
Download a llama_cpp model (gguf format) appropriate for your machine.
E.g. on [https://huggingface.co/TheBloke/Stable-Platypus2-13B-GGUF]
you can see the RAM requirements dependent on the quantization method
-> Tradeoff: RAM & Performance vs Quality
## Run demo

``` python demo/main.py ```

