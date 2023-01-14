from defake.paths import coco_annotations_train2017_path, coco_annotations_val2017_path, annotations_train_path, annotations_val_path
import json
from tqdm import tqdm



def create_annotations_file(input_file, output_file):
    with open(input_file) as json_file:
        captions_json = json.loads(json_file.read())
    
    images = captions_json['images']
    annotations = captions_json['annotations']
    
    
    samples_dict = {}
    
    for image in tqdm(images):
        image_id = image['id']
        
        # Find annotations
        captions = []
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                captions.append(annotation['caption'])
        
        
        image_dict = {'file_name': image['file_name'],
                      'coco_url': image['coco_url'],
                      'flickr_url': image['flickr_url'],
                      'height': image['height'],
                      'width': image['width'],
                      'captions': captions}
        
        samples_dict[image_id] = image_dict
        
    with open(output_file, 'w') as f:
        json.dump(samples_dict, f)
    
    


def main():
    create_annotations_file(coco_annotations_val2017_path, annotations_val_path)
    create_annotations_file(coco_annotations_train2017_path, annotations_train_path)


if __name__ == '__main__':
    main()




