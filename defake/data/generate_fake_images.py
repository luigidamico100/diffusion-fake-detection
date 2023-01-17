from defake.paths import (annotations_train_path, annotations_val_path,
                         coco_train2017_dir, coco_val2017_dir,
                         dataset_generated_train_dir, dataset_generated_val_dir,
                         dataset_real_train_dir, dataset_real_val_dir,
                         dataset_annotations_train_path, dataset_annotations_val_path)


import json
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from defake.config import device


pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(device) # https://sebastianraschka.com/blog/2022/pytorch-m1-gpu.html
pipe.enable_attention_slicing()
_ = pipe('prova', num_inference_steps=1)

num_inference_steps = 50


def generate_fake_images(pipe, real_images_dir, annotations_path, dataset_generated_images_dir, dataset_real_images_dir, annotations_generated_path, num_inference_steps=50):    

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
        real_image_name = annotation['file_name']
        prompt = annotation['captions'][0]
        
        nsfw_content = True
        while nsfw_content:
            result = pipe(prompt, num_inference_steps=num_inference_steps)
            # Try again for not NSFW generated content
            # https://huggingface.co/blog/stable_diffusion
            generated_image = result.images[0]
            
            nsfw_content = result['nsfw_content_detected'][0]
            
            if nsfw_content:
                print(f'Generated NSFW content for image {real_image_name}. Trying again')
        
        generated_image_name = 'SD15_' + real_image_name
        generated_image_path = os.path.join(dataset_generated_images_dir, generated_image_name)
        generated_image.save(generated_image_path)
        print(f'Generated image saved to: {generated_image_path}')
        
        real_image_path = os.path.join(real_images_dir, real_image_name)
        real_image = Image.open(real_image_path)
        real_image_path = os.path.join(dataset_real_images_dir, real_image_name)
        real_image.save(real_image_path)
        print(f'Real image saved to: {real_image_path}')
        
        
        
        annotations_generated_dict[image_id] = {
            'image_name': real_image_name,
            'generated_image_name': generated_image_name,
            'prompt': prompt}
        
        with open(annotations_generated_path, 'w') as f:
            json.dump(annotations_generated_dict, f)


    
def generate_train_images():
    generate_fake_images(pipe, 
                         real_images_dir=coco_train2017_dir,  
                         annotations_path=annotations_train_path,
                         dataset_generated_images_dir=dataset_generated_train_dir,
                         dataset_real_images_dir=dataset_real_train_dir,
                         annotations_generated_path=dataset_annotations_train_path,
                         num_inference_steps=num_inference_steps)
    
def generate_val_images():
    generate_fake_images(pipe, 
                         real_images_dir=coco_val2017_dir,  
                         annotations_path=annotations_val_path,
                         dataset_generated_images_dir=dataset_generated_val_dir,
                         dataset_real_images_dir=dataset_real_val_dir,
                         annotations_generated_path=dataset_annotations_val_path,
                         num_inference_steps=num_inference_steps)
    

if __name__ == '__main__':
    generate_val_images()
    
    
    
    
    
    
    