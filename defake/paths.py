import os
from pathlib import Path



project_root_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)
data_root_path = '/Users/gigi/NoCloud'

# real images from coco
coco_annotations_dir = os.path.join(data_root_path, 'data', 'real_images','annotations')
coco_annotations_train2017_path = os.path.join(coco_annotations_dir, 'captions_train2017.json')
coco_annotations_val2017_path = os.path.join(coco_annotations_dir, 'captions_val2017.json')
coco_train2017_dir = os.path.join(data_root_path, 'data', 'real_images', 'train2017')
coco_val2017_dir = os.path.join(data_root_path, 'data', 'real_images', 'val2017')

# generated annotations: images <-> captions
annotations_dir = os.path.join(data_root_path, 'data', 'annotations')
annotations_train_path = os.path.join(annotations_dir, 'annotations_train.json')
annotations_val_path = os.path.join(annotations_dir, 'annotations_val.json')

# generated images
generated_images_path = os.path.join(data_root_path, 'data', 'generated_images')
generated_train_dir = os.path.join(generated_images_path, 'train')
generated_val_dir = os.path.join(generated_images_path, 'val')
generated_annotations_train_path = os.path.join(generated_images_path, 'annotations', 'annotations_generated_train.json')
generated_annotations_val_path = os.path.join(generated_images_path, 'annotations', 'annotations_generated_val.json')


    
