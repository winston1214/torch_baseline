from dataset import ScalpDataset,Proces_dataset
from model import StyleModel
import torch
import argparse
import numpy as np
import timm
import argparse
import torchvision.transforms as T
import torch.utils.data as D
from tqdm import tqdm

def test(opt):
    device = 'cuda:0'
    pretrained = timm.create_model('convnext_base')
    model = StyleModel(pretrained)
    model.load_state_dict(torch.load(opt.ckpt))
    model.to(device)
    processing = Proces_dataset(opt.path, 'instances_default.json')
    test_img,test_label = processing.train_val_test('/data/dmc/3_dota','test')
    test_transformer = T.Compose([T.ToTensor()])

    test_dataset = ScalpDataset(opt.path+'images/', test_img,test_label,transform = test_transformer)
    test_dataloader = D.DataLoader(test_dataset, batch_size = opt.batch, shuffle = False, drop_last=False)
    prediction_list = np.array([])

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            probs = model(images)
            
            probs = probs.cpu().detach().numpy()
            preds = probs > 0.5
            prediction_list = np.append(prediction_list,preds.astype(np.int)).reshape(-1,8)
    return prediction_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str,default='/data/dmc/2_coco/')
    parser.add_argument('--ckpt',type=str)
    parser.add_argument('--batch',type=int, default= 16)
    opt = parser.parse_args()
    pred = test(opt)
    print(pred)

