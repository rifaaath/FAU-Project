import os
import torch
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import frcnn.transforms as T
import frcnn.utils as utils
from torchvision import transforms
from training import HomerCompDataset
from tqdm import tqdm

mapping = {
    1: 7,
    2: 8,
    3: 9,
    4: 14,
    5: 17,
    6: 23,
    7: 33,
    8: 45,
    9: 59,
    10: 77,
    11: 100,
    12: 107,
    13: 111,
    14: 119,
    15: 120,
    16: 144,
    17: 150,
    18: 161,
    19: 169,
    20: 177,
    21: 186,
    22: 201,
    23: 212,
    24: 225,
}



device="cpu"
num_classes=25

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

checkpoint = torch.load("model_detection.pt", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

trans = []
trans.append(T.PILToTensor())
trans.append(T.ConvertImageDtype(torch.float))
trans = T.Compose(trans)

dataset_test = HomerCompDataset(transforms=trans, isTrain=False)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)



jFile = open(os.path.join("template.json"))
json_output = json.load(jFile)
jFile.close()

jFile = open(os.path.join("HomerCompTraining", "HomerCompTrainingReadCoco.json"))
test_gt = json.load(jFile)
jFile.close()

img_ids = []

def main():
    for images, targets in tqdm(data_loader_test):
        
        image = images[0]
        idx = targets[0]['image_id'].item()

        image_id = json_output['images'][dataset_test.imgs[idx]]['bln_id']
        img_ids.append(image_id)


        # Patch wise predictions    
        for i in range(0, image.shape[1], 672):
            for j in range(0, image.shape[2], 672):
                crop = transforms.functional.crop(image, i, j, 672, 672)
                crop = torch.unsqueeze(crop, 0)
                result = model(crop)
                boxes = result[0]['boxes'].int()
                scores = result[0]['scores']
                preds = result[0]['labels']
                if len(boxes) == 0:
                    continue
            

                for box, label, score in zip(boxes, preds, scores) :
                    annotation = dict()
                    annotation['image_id'] = image_id
                    annotation['category_id'] = mapping[label.item()]
                    annotation['bbox'] = [box[0].item() + j, box[1].item() + i, (box[2] - box[0]).item(), (box[3] - box[1]).item()]
                    annotation['score'] = score.item()

                    json_output['annotations'].append(annotation)


    for annotation in list(test_gt['annotations']) :
        if annotation['image_id'] not in img_ids :
            test_gt['annotations'].remove(annotation)

    with open("gt.json", "w") as outfile:
        json.dump(test_gt, outfile, indent=4) 

    with open("predictions.json", "w") as outfile:
        json.dump(json_output, outfile, indent=4) 

if __name__ == '__main__':
    main()
