import os
from pathlib import Path



project_root_path = os.path.dirname(Path(os.path.abspath(__file__)).parent)

# real images from coco
coco_annotations_dir = os.path.join(project_root_path, 'data', 'real_images','annotations')
coco_annotations_train2017_path = os.path.join(coco_annotations_dir, 'captions_train2017.json')
coco_annotations_val2017_path = os.path.join(coco_annotations_dir, 'captions_val2017.json')
coco_train2017_dir = os.path.join(project_root_path, 'data', 'real_images', 'train2017')
coco_val2017_dir = os.path.join(project_root_path, 'data', 'real_images', 'val2017')

# generated annotations: images <-> captions
annotations_dir = os.path.join(project_root_path, 'data', 'annotations')
annotations_train_path = os.path.join(annotations_dir, 'annotations_train.json')
annotations_val_path = os.path.join(annotations_dir, 'annotations_val.json')

# # generated images
# generated_images_path = os.path.join(project_root_path, 'data', 'generated_images')
# generated_train_dir = os.path.join(generated_images_path, 'train')
# generated_val_dir = os.path.join(generated_images_path, 'val')
# generated_annotations_train_path = os.path.join(generated_images_path, 'annotations', 'annotations_generated_train.json')
# generated_annotations_val_path = os.path.join(generated_images_path, 'annotations', 'annotations_generated_val.json')

# dataset images
dataset_path = os.path.join(project_root_path, 'data', 'dataset')
dataset_generated_train_dir = os.path.join(dataset_path, 'generated_images', 'train')
dataset_generated_val_dir = os.path.join(dataset_path, 'generated_images', 'val')
dataset_real_train_dir = os.path.join(dataset_path, 'real_images', 'train')
dataset_real_val_dir = os.path.join(dataset_path, 'real_images', 'val')
dataset_annotations_train_path = os.path.join(dataset_path, 'annotations', 'dataset_annotations_train.json')
dataset_annotations_val_path = os.path.join(dataset_path, 'annotations', 'dataset_annotations_val.json')

# logs
runs_path = os.path.join(project_root_path, 'runs')





