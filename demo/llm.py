import os
import gradio as gr
import numpy as np
from llama_cpp import Llama, StoppingCriteriaList
from functools import partial

device = "cpu"


# Has to be a GGUF model since llama-cpp-python 0.1.78+
# Currently https://huggingface.co/TheBloke/Stable-Platypus2-13B-GGUF/blob/main/stable-platypus2-13b.Q4_K_M.gguf
MODEL_NAME = "model.gguf"

PROMPT_TEMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{user_input}

### Response:
"""

bot = None
eos_token = None
top_tokens: dict[int, list[tuple[int, float]]] = {}
last_prompt = ""


def prepare_chat(user_input: str):
    return [(user_input, None)], ""


def stop_on_high_eos_logit(input_ids: list[int], logits: list[float]) -> bool:
    eos_p = np.exp(logits[eos_token]) / np.sum(np.exp(np.asarray(logits)), axis=0)
    if eos_p > 0.1:
        print(f"STOP: P[eos] = {eos_p} > 0.1")
        return True
    print(f"CONTINUE: P[eos] = {eos_p} <= 0.1")
    return False


def get_top_n_logits(input_ids: list[int], logits: list[float], n=5):
    global top_tokens
    ind = np.argpartition(logits, -n)[-n:]
    i = len(top_tokens)
    top_tokens[i] = []
    for token in ind:
        p = np.exp(logits[token]) / np.sum(np.exp(np.asarray(logits)), axis=0)
        top_tokens[i].append((token, p))
    return logits


top_5_tokens_processor = partial(get_top_n_logits, n=5)


def respond(bot_history, bot: Llama):
    print("respond")
    global last_prompt
    prompt = PROMPT_TEMPLATE.format(user_input=bot_history[-1])
    encoded_prompt = bytes(prompt, encoding="utf-8")
    tokens = bot.tokenize(encoded_prompt)
    bot_history[-1][1] = ""
    stopping_criteria = StoppingCriteriaList([stop_on_high_eos_logit])
    for token in bot.generate(
        tokens, temp=0.1, top_k=10, stopping_criteria=stopping_criteria
    ):
        output_bytes = bot.detokenize([token])
        bot_history[-1][1] += bytes.decode(output_bytes)
        yield bot_history


# TODO: There is a (de-)serialization bug
# In particular newline tokens `\n` will not be correctly displayed in the widget
# Further a selection of such a token might cause the program to crash
def _generate_suggestions(bot: Llama):
    global top_tokens
    best_tokens = top_tokens[len(top_tokens) - 1]
    suggestions = []
    best_tokens = sorted(best_tokens, key=lambda xs: xs[1], reverse=True)
    top_tokens[len(top_tokens)] = best_tokens
    for token_id, probability in best_tokens:
        token = bytes.decode(bot.detokenize([token_id]), encoding="utf-8")
        # token = bytes.decode(bot.detokenize([token_id]))
        # token = bot.detokenize([token_id])
        suggestions.append(f"'{token}' - probability: {round(probability, 3)}")
    # print(suggestions)
    return suggestions


def generate_list_of_probable_tokens(bot_history, bot: Llama):
    global last_prompt
    prompt = PROMPT_TEMPLATE.format(user_input=bot_history[-1])
    last_prompt = prompt
    # encoded_prompt = bytes(prompt, encoding="utf-8")
    # tokens = bot.tokenize(encoded_prompt)
    bot.create_completion(
        prompt=prompt, logits_processor=top_5_tokens_processor, max_tokens=1
    )
    suggestions = _generate_suggestions(bot)
    return gr.CheckboxGroup.update(choices=suggestions, value=None)


def generate_list_of_probable_tokens_from_selection(bot_history, selection, bot: Llama):
    token = selection[0].split("' - probability")[0][1:]
    print(f"Picked token {token} from `{selection}`")
    global last_prompt
    last_prompt += token
    bot.create_completion(
        prompt=last_prompt, logits_processor=top_5_tokens_processor, max_tokens=1
    )
    suggestions = _generate_suggestions(bot)
    if bot_history[-1][1] is None:
        bot_history[-1][1] = ""
    bot_history[-1][1] += token
    if token in ["", eos_token]:
        return bot_history, gr.CheckboxGroup.update(choices=[], value=None)
    return bot_history, gr.CheckboxGroup.update(choices=suggestions, value=None)


def continue_with_user_input(bot_history, user_input, bot: Llama):
    global last_prompt
    last_prompt += user_input
    if bot_history[-1][1] is None:
        bot_history[-1][1] = ""
    bot_history[-1][1] += user_input

    bot.create_completion(
        prompt=last_prompt, logits_processor=top_5_tokens_processor, max_tokens=1
    )
    suggestions = _generate_suggestions(bot)
    return "", bot_history, gr.CheckboxGroup.update(choices=suggestions, value=None)


EXAMPLES = [
    ['What is the past tense of "I am"?'],
    ["What is Creative Approval?"],
    ["""Answer the following question based on the information provided in the context block.

Question:
What is the Creative Approval?

Context:
Creative Approval is a smartclip tool that allows publishers to approve and reject the advertisements that run on their inventory, giving them more granular control to protect their brand. Creative Approval is especially important for sensitive brands that have strict rules they must follow when it comes to what can be shown on their sites."""]
]


def attach_next_token_prediction(app: gr.Blocks, bot):
    with gr.Tab("LLM - Next token"):
        gr.Markdown("# Predict tokens one at a time")
        with gr.Row():
            with gr.Column(scale=1.0):
                chat_history = gr.Chatbot([], elem_id="chat_history")
        user_input = gr.Textbox(
            label="User input", placeholder="Input", elem_id="user_input"
        )

        next_token_selector = gr.CheckboxGroup(label="Next token", elem_id="next_token")
        user_continuation = gr.Textbox(
            label="User continunation",
            placeholder="Custom continuation",
            elem_id="user_cont",
        )

        bot_generator = partial(generate_list_of_probable_tokens, bot=bot)
        selection_generator = partial(
            generate_list_of_probable_tokens_from_selection, bot=bot
        )

        continue_from_user_input_generator = partial(continue_with_user_input, bot=bot)

        chatbot_after_input = user_input.submit(
            fn=prepare_chat,
            inputs=user_input,
            outputs=[chat_history, user_input],
            queue=False,
        )
        chatbot_after_input.then(
            bot_generator, inputs=chat_history, outputs=next_token_selector
        )

        next_token_selector.input(
            fn=selection_generator,
            inputs=[chat_history, next_token_selector],
            outputs=[chat_history, next_token_selector],
        )

        user_continuation.submit(
            fn=continue_from_user_input_generator,
            inputs=[chat_history, user_continuation],
            outputs=[user_continuation, chat_history, next_token_selector],
        )

        gr.Examples(
            EXAMPLES,
            inputs=user_input,
            outputs=user_input,
        )


def attach_normal_prediction(app: gr.Blocks, bot):
    with gr.Tab("LLM - Normal"):
        gr.Markdown("# generate text")
        chat_history = gr.Chatbot([])

        user_input = gr.Textbox(label="User input", placeholder="Input")

        bot_generator = partial(respond, bot=bot)
        user_input.submit(
            fn=prepare_chat,
            inputs=user_input,
            outputs=[chat_history, user_input],
            queue=False,
        ).then(bot_generator, inputs=chat_history, outputs=chat_history)

        gr.Examples(
            EXAMPLES,
            inputs=user_input,
            outputs=user_input,
        )


def attach(app: gr.Blocks):
    project_dir = "/".join(__file__.split("/")[:-2])
    model_path = f"{project_dir}/models/{MODEL_NAME}"
    assert os.path.isfile(model_path), f"Model {model_path} doesn't exist"
    bot = Llama(model_path)
    global eos_token
    eos_token = bot.token_eos()
    attach_next_token_prediction(app, bot)
    attach_normal_prediction(app, bot)
