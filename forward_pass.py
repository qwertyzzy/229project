import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import data
from tqdm import tqdm


MODEL_NAME = 'resnet'


class ImageDSet(torch.utils.data.Dataset):
    def __init__(self, fnames):
        self.fnames = fnames

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
        image = transform(image)
        return image


def main():
    model_name = MODEL_NAME

    dataloader = data.DataLoader('/lfs/1/zhyzhang/yt-videos/yt_bb_detection_validation/', '../cluster.db')
    image_paths = dataloader.get_image_paths()

    dset = ImageDSet(image_paths)
    loader = torch.utils.data.DataLoader(
            dset, shuffle=False,
            batch_size=1, num_workers=16
    )

    if model_name == 'vgg':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
    elif model_name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    model.cuda()
    model.eval()

    outputs = []
    with torch.no_grad():
        for idx, batch  in enumerate(tqdm(loader)):
            batch = batch.cuda(non_blocking=True)
            out = model(batch)
            outputs.append(out[0].cpu().numpy())
    
    print(np.array(outputs).shape)
    outputs = np.squeeze(outputs)
    print(np.array(outputs).shape)
    np.savetxt('{}_final_weights.txt'.format(model_name), np.array(outputs))
        

if __name__ == "__main__":
    main()
