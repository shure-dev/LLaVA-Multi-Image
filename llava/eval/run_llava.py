import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open("/content/LLaVA-Multi-Image/"+image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # qs = args.query
    image_files = [
        "vima-images/task-14/task-14-prompt-object-01.png", # {object image for target task}
        "vima-images/task-14/task-14-current-snence.png", # {curent scene for target task}
        
        "vima-images/task-01/task-01-prompt-object-01.png",  # {first object image for example task 1}
        "vima-images/task-01/task-01-prompt-object-02.png",  # {first object image for example task 1}
        "vima-images/task-01/task-01-context.png",  # {first object image for example task 1}

        "vima-images/task-11/task-11-scene-01.png",  # {first object image for example task 1}
        "vima-images/task-11/task-11-scene-02.png",  # {first object image for example task 1}
        "vima-images/task-11/task-11-scene-03.png",  # {first object image for example task 1}
        "vima-images/task-11/task-11-current-scene.png",  # {first object image for example task 1}

        "vima-images/task-02/task-02-prompt-scene-01.png",  # {first object image for example task 1}
        "vima-images/task-02/task-02-current-scene.png",  # {first object image for example task 1}

        ]
    
    qs = '''
    
    # This is description for given 11 images
    Image 1: {object image for target task}
    Image 2: {curent scene for target task}
    
    Image 3: {first object image for example task 1}
    Image 4: {second object image for example task 1}
    Image 5: {curent scene for example task 1}
    
    Image 6: {first scene for example task 2}
    Image 7: {second scene for example task 2}
    Image 8: {third scene for example task 2}
    Image 9: {curent scene for example task 2}

    Image 10: {scene for example task 3}
    Image 11: {curent scene for example task 3}

    # Decompose this target task, following examples
    Target task: Put all objects with the same texture as <image-placeholder> {object image for target task} into it in this scene <image-placeholder> {curent scene for target task}.
    
    # Examples
    ## Task1
    Original: Put the <image-placeholder> {first object image for example task 1} into the <image-placeholder> {second object image for example task 1} in this scene <image-placeholder> {curent scene for example task 1}.
    After decomposition: 
    Pick the R-shaped object and place it in the green container.

    ## Task2
    Original: Stack objects in this order <image-placeholder>{first scene for example task 2} <image-placeholder>{second scene for example task 2}<image-placeholder>{third scene for example task 2} in this scene <image-placeholder> {curent scene for example task 2}.
    After decomposition:
    Pick the green stripe & triangle object and place it in the gray & triangle object.
    Pick the rainbow & triangle object and place it in the green stripe & triangle object.
    
    ## Task3
    Original: Put the green and blue stripe object in <image-placeholder> {scene for example task 3} into the yellow paisely object in this scene <image-placeholder> {curent scene for example task 3}.
    After decomposition:
    Pick the green and blue stripe object and place it in the yellow paisely object.

    '''
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image_files = image_parser(args)
    images = load_images(image_files)
    
    images_tensor = (
        image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    args = parser.parse_args()

    eval_model(args)
