from defake.paths import (annotations_train_path, annotations_val_path, coco_train2017_dir, coco_val2017_dir, 
                          generated_val_dir, generated_annotations_val_path, generated_annotations_train_path,
                          generated_train_dir)
import json
import os
from diffusers import StableDiffusionPipeline
import torch


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("mps") # https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html
pipe.enable_attention_slicing()
_ = pipe('prova', num_inference_steps=1)

num_inference_steps = 50


def generate_fake_images(pipe, real_images_dir, generated_images_dir, annotations_path, annotations_generated_path, num_inference_steps=50):

    annotations_generated_dict = {}
    
    with open(annotations_path) as json_file:
        annotations = json.loads(json_file.read())
        
    try: 
        with open(annotations_generated_path) as json_file:
            annotations_generated_dict = json.loads(json_file.read())
            print('---- Annotations_generated json correctly loaded')
    except FileNotFoundError:
        # First time you are generated these images
        print('---- First time you are generated these images ----')
        pass
        
    
    for image_id in annotations:
        
        if image_id in annotations_generated_dict:
            print(f'image id: {image_id}, already generated')
            continue
        
        annotation = annotations[image_id]
        image_name = annotation['file_name']
        prompt = annotation['captions'][0]
        result = pipe(prompt, num_inference_steps=num_inference_steps)
        # Try again for not NSFW generated content
        # https://huggingface.co/blog/stable_diffusion
        image = result.images[0]
        
        generated_image_name = 'SD15_' + image_name
        generated_image_path = os.path.join(generated_images_dir, generated_image_name)
        image.save(generated_image_path)
        print(f'Image saved to: {generated_image_path}')
        
        annotations_generated_dict[image_id] = {
            'image_name': image_name,
            'generated_image_name': generated_image_name,
            'prompt': prompt}
        
        with open(annotations_generated_path, 'w') as f:
            json.dump(annotations_generated_dict, f)



generate_fake_images(pipe, 
                     real_images_dir=coco_train2017_dir, 
                     generated_images_dir=generated_train_dir, 
                     annotations_path=annotations_train_path,
                     annotations_generated_path=generated_annotations_train_path,
                     num_inference_steps=num_inference_steps)