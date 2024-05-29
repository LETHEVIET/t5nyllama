import gradio as gr
import errant
import spacy
import os
import json
import nltk
from utils import get_random_prompt, instruction_prompts
from llama_cpp import Llama
from transformers import pipeline
import config

# Load necessary models and resources
nlp = spacy.load("en_core_web_sm")
annotator = errant.load('en', nlp)
errant_path = os.path.join(os.path.dirname("./"), 'errant_verbose.json')
errant_verbose = json.load(open(errant_path, "r"))
sent_detector = nltk.data.load('./nltk_data/tokenizers/punkt/english.pickle')
print("Loading models ...")
# Load text editor (TinyLlama)
text_editor = Llama(
    model_path="./texteditor-model/coedit-tinyllama-chat-bnb-4bit-unsloth.Q4_K_M.gguf",
    verbose=True
)
print("text editor is loaded!")

# Load grammar corrector (Flan-T5)
grammar_corrector = pipeline(
    'text2text-generation',
    'pszemraj/flan-t5-large-grammar-synthesis',
)
print("grammar corrector is loaded!")

def correcting_text(src: str) -> str:
    """
    Corrects grammatical errors in the given text using the grammar corrector model.

    Args:
        src: The text to be corrected.

    Returns:
        The grammatically corrected text.
    """
    lines = src.split('\n')
    sentences = []
    line_idx = []
    for l_idx, line in enumerate(lines):
        if len(line) == 0:
            continue
        l_sents = sent_detector.tokenize(line)
        for sent in l_sents:
            sentences.append(sent)
            line_idx.append(l_idx)

    num_iter = (len(sentences) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    final_outs = []
    out_lines = ["" for _ in lines]
    for i in range(num_iter):
        start = i * config.BATCH_SIZE
        end = min((i + 1) * config.BATCH_SIZE, len(sentences))
        
        final_outs += grammar_corrector(sentences[start:end], max_length=128, num_beams=5, early_stopping=True)
        

    for i in range(len(final_outs)):
        out_lines[line_idx[i]] += final_outs[i]["generated_text"] + " " 

    return "\n".join(out_lines)

def annotate_text(src: str, tag: str, analyze: bool = True) -> list:
    """
    Annotates the text with edits based on the provided tag using the Errant library.
    original code from: https://github.com/nusnlp/ALLECS
    Args:
        src: The source text.
        tag: The target text.
        analyze: Whether to analyze and provide detailed information about edits.

    Returns:
        A list of tuples representing the edits, where each tuple is:
        - (edit_text, edit_type)
    """
    out = {"edits": []}
    out['source'] = src
    src_doc = annotator.parse(src)
    tag_doc = annotator.parse(tag)
    cur_edits = annotator.annotate(src_doc, tag_doc)

        
    for e in cur_edits:
        out["edits"].append((e.o_start, e.o_end, e.type, e.c_str))
    result = []
    last_pos = 0
    if analyze:
        tokens = out['source']
        if isinstance(tokens, str):
            tokens = tokens.split(' ')
        edits = out['edits']
        offset = 0
        for edit in edits:
            if isinstance(edit, dict):
                e_start = edit['start']
                e_end = edit['end']
                e_type = edit['type']
                e_rep = edit['cor']
            elif isinstance(edit, tuple):
                e_start = edit[0]
                e_end = edit[1]
                e_type = edit[2]
                e_rep = edit[3]
            else:
                raise ValueError("Data type {} is not supported."\
                        .format(type(edit)))

            e_rep = e_rep.strip()
            op_type = e_type[0]
            pos_type = e_type[2:]
            errant_info = errant_verbose[pos_type]
            title = errant_info["title"]
            
            result.append((' '.join(tokens[last_pos:e_start + offset]), None))
            
            ori_str = ' '.join(tokens[e_start + offset:e_end + offset]).strip()
            if pos_type == "ORTH":
                # check if it's a casing issue
                if ori_str.lower() == e_rep.lower():
                    if e_rep[0].isupper() and ori_str[0].islower():
                        msg = "<b>{ori}</b> should be capitalized."
                    elif e_rep[0].islower() and ori_str[0].isupper():
                        msg = "<b>{ori}</b> should not be capitalized."
                    else:
                        msg = "The casing of the word <b>{ori}</b> is wrong."
                # then it should be a spacing issue
                else:
                    if len(ori_str) - 1 == len(e_rep):
                        msg = "The word <b>{ori}</b> should not be written separately."
                    elif len(ori_str) + 1 == len(e_rep):
                        msg = "The word <b>{ori}</b> should be separated into <b>{cor}</b>."
                    else:
                        msg = "The word <b>{ori}</b> has orthography error."
            else:
                if op_type in errant_info:
                    msg = errant_info[op_type]
                else:
                    msg = errant_verbose["Default"][op_type]
            
            msg = '<p>' + msg.format(ori=ori_str, cor=e_rep) + '</p>'

            e_cor =  e_rep.split()
            len_cor = len(e_cor)
            tokens[e_start + offset:e_end + offset] = e_cor
            last_pos = e_start + offset + len_cor 
            offset = offset - (e_end - e_start) + len_cor
            result.append((e_rep, pos_type))
        out = ' '.join(tokens)
        result.append((' '.join(tokens[last_pos:]), None))
    print(result)
    return result

def choices2promts() -> list:
    """
    Returns a list of available instructions for text editing.

    Returns:
        A list of instruction names.
    """
    return instruction_prompts.keys()

with gr.Blocks() as demo: 

    def turn_off_legend(msg: str) -> gr.update:
        """
        Turns off the legend in the highlighted text component.

        Args:
            msg: The text input.

        Returns:
            A Gradio update object to hide the legend.
        """
        return gr.update(show_legend=False)

    def turn_on_legend(annotate: bool) -> gr.update:
        """
        Turns on the legend in the highlighted text component if annotate is True.

        Args:
            annotate: Whether to show annotations.

        Returns:
            A Gradio update object to show or hide the legend.
        """
        if annotate:
            return gr.update(show_legend=True)
        else:
            return gr.update(show_legend=False)

    def bot(task: str, text: str, post_check: bool, annotate: bool) -> tuple:
        """
        Processes the user input and returns the edited text along with annotations.

        Args:
            task: The chosen instruction for editing.
            text: The text to be edited.
            post_check: Whether to check for grammatical errors after text generation.
            annotate: Whether to show annotations.

        Yields:
            Tuples of (edited text, annotation type) to update the interface.
        """
        response = ""
        if task == "Grammar Error Correction":
            yield [("Processing ...", None)], "Checking Grammar ..."
            response = correcting_text(text)
        else:
            instruction = get_random_prompt(task)
            prompt = instruction + ": " + text
            print(prompt)
            output = text_editor.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an English writing assistant, editing the text of user input and response based on user instructions. Please do not provide explanations, but respond only with the edited text. Also, if the instruction is not provided, correct the grammar of the text. Finally, if the instruction is not for editing text, correct the grammar of the text.",
                    },
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=0.0,
                stream=True,
            )

            response = ""
            for chunk in output:
                delta = chunk["choices"][0]["delta"]
                if "role" in delta:
                    pass
                elif "content" in delta:
                    response+=delta['content']
                    res = [(response, None), ]
                    print(res)
                    yield res, "Generating output ..."
            
            if post_check:
                yield [(response, None)], "Checking Grammar ..."
                response = correcting_text(response)

        print(response)

        if annotate:
            e_edit = annotate_text(text, response)
        else:
            e_edit = [(response, None)]
        
        yield e_edit, "Done."

    def handle_highlight_selection():
        """
        Handles the selection event of the highlighted text component.

        This function is not implemented in the original code.
        """
        # print("hi")
        return

    gr.Markdown("# English Text Editing Application using T5 and Tiny Llama")
    gr.Markdown("> source code: https://github.com/LETHEVIET/t5nyllama")
    with gr.Row() as row:
        with gr.Column(scale=1) as col1:
            instruction = gr.Dropdown(
                choices=choices2promts(),
                value="Grammar Error Correction",
                multiselect=False,
                label="Choose your instruction",
                interactive=True,
                scale=0
            )

            with gr.Row() as row2:
                clear = gr.Button("Clear", scale=-1)
                submit = gr.Button("submit", scale=-1)
                
            info_msg = gr.Textbox(
                label="Information",
                scale=1,
                lines=3,
                value="Information will show here.",
            )

            post_check = gr.Checkbox(label="Check grammaticality after text generation.", value=True)
            annotate = gr.Checkbox(label="Highlight different", value=True)
        with gr.Column(scale=2) as col2:
            msg = gr.Textbox(
                label="Input",
                scale=3,
                value="i can has cheezburger.",
            )

            result = gr.HighlightedText(
                label="Result",
                combine_adjacent=True,
                show_legend=False,
                scale=3
            )
        
    res_msg = gr.Textbox(
        scale=0,
        visible=False,
        label="Ouput",
    )
    
    msg.submit(turn_off_legend, msg, result).then(bot, [instruction, msg, post_check, annotate], [result, info_msg]).then(turn_on_legend, annotate, result)

    clear.click(lambda: None, None, result, queue=False)

    submit.click(turn_off_legend, msg, result).then(bot, [instruction, msg, post_check, annotate], [result, info_msg]).then(turn_on_legend, annotate, result)

    result.select(handle_highlight_selection, [], [])

if __name__ == "__main__":
    demo.launch(server_port=7860)