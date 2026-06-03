"""
@author: harib
"""
from networks import ContentEncoder, ReparameterizedStyleEncoder, DecoderConcat, AdaINDecoder, Discriminator, ContentDiscriminator
from dataset import dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import os, sys, gc
from PIL import Image
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import argparse

class Transform2Neutral():
    def __init__(self, root, neutral_template_path, target_path, model, weight_path,
                 batch_size=1, height=720, width=1280, keep_resolution=False, num_domains=5, input_dim=3,
                 hot_vector=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.root = root
        self.neutral_template_path = neutral_template_path
        self.target_path = target_path
        self.model = model
        self.batch_size = batch_size
        self.num_domains = num_domains
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.device = device
        self.weight_path = weight_path
        self.keep_resolution = keep_resolution

        if hot_vector is None:
            hot_vector = torch.ones(num_domains)

        self.hot_vector = hot_vector
        
    def save_weight(self):
        network_name = ['self.content_encoder', 
                        'self.style_encoder', 
                        'self.decoder']
        
        network = [ContentEncoder(self.input_dim), 
                   ReparameterizedStyleEncoder(self.input_dim, num_domains = self.num_domains, activation='lrelu')] 

        if self.model == 'base_model':
            network.insert(2, DecoderConcat(self.input_dim, num_domains=self.num_domains))
        if self.model == 'adain_model':
            network.insert(2, AdaINDecoder(self.input_dim, num_domains=self.num_domains))

        all_models = torch.load(self.weight_path, map_location=torch.device("cpu"))
        for i in range(len(network_name)):
            weights = all_models[network_name[i].split('.')[1]]
            new_state_dict = OrderedDict()
            for k, v in weights.items():
                name = k[7:]
                new_state_dict[name] = v
            network[i].load_state_dict(new_state_dict)
            network[i].to(self.device)
            exec(f"{network_name[i]}  = network[i]")
    
    def set_inputs(self, inputs):
        img_a = inputs['x1'].to(self.device).detach()
        cls_a = inputs['y1'].to(self.device).detach()
        img_b = inputs['x2'].to(self.device).detach()
        cls_b = inputs['y2'].to(self.device).detach()
        name = inputs['name']
        img = torch.cat((img_a, img_b), dim=0)
        c_org = torch.cat((cls_a, cls_b), dim=0)
        return img, c_org, name
    
    def image_transform(self, img, c_org):
        self.save_weight()
        z_c = self.content_encoder(img)
        z_ca, z_cb = torch.split(z_c, self.batch_size, dim=0)
        z_s, mu, logvar = self.style_encoder(img, c_org)
        #z_s = style_encoder(img, c_org)
        z_sa, z_sb = torch.split(z_s, self.batch_size, dim=0)
        z_sr = torch.randn(1, 8).to(self.device)
    
        cls_a, cls_b = torch.split(c_org, self.batch_size, dim=0)
        # B -> A
        content = torch.cat((z_cb, z_ca, z_cb), dim=0)
        style = torch.cat((z_sa, z_sa, z_sr), dim=0)
        trg_cls = torch.cat((cls_a, cls_a, cls_a), dim=0)
        fake_imgs = self.decoder(content, style, trg_cls)
        img_ba, img_aa, img_br = torch.split(fake_imgs, self.batch_size, dim=0)
        
        # A -> B
        content = torch.cat((z_ca, z_cb, z_ca), dim=0)
        style = torch.cat((z_sb, z_sb, z_sr), dim=0)
        trg_cls = torch.cat((cls_b, cls_b, cls_b), dim=0)
        fake_imgs = self.decoder(content, style, trg_cls)
        img_ab, img_bb, img_ar = torch.split(fake_imgs, self.batch_size, dim=0)
        # concatinate images
        img_fake = torch.cat((img_ba, img_ab), dim=0)
        fake_img1, fake_img2 = torch.split(img_fake, self.batch_size, dim=0)
        return fake_img1, fake_img2
    
    def run(self):
        data = dataset(root = self.root,
                       neutral_template_path = self.neutral_template_path,
                       num_domains = self.num_domains, height = self.height, width = self.width,
                       hot_vector=self.hot_vector, keep_resolution=self.keep_resolution)
        
        dataloader = DataLoader(data, batch_size=self.batch_size,
                                num_workers= 4, shuffle=True)
        
        for it, batch in enumerate(dataloader):
            image, c_org, name = self.set_inputs(batch)
            with torch.no_grad():
                _, fake_img2 = self.image_transform(image, c_org)
            if os.path.isdir('%s/%s/%s'%(self.target_path, self.model, name[0].split('\\')[-2])):
                save_image(fake_img2, '%s/%s/%s/%s.png'%(self.target_path, self.model, name[0].split('\\')[-2], name[0].split('\\')[-1].split('.')[0]))
            else:
                os.makedirs('%s/%s/%s'%(self.target_path, self.model, name[0].split('\\')[-2]))
                save_image(fake_img2, '%s/%s/%s/%s.png'%(self.target_path, self.model, name[0].split('\\')[-2], name[0].split('\\')[-1].split('.')[0]))
            torch.cuda.empty_cache()
            
def parse_hot_vector(value, num_domains):
    if value == "ones":
        return torch.ones(num_domains)

    if value == "zeros":
        return torch.zeros(num_domains)

    values = [float(x) for x in value.split(",")]

    if len(values) != num_domains:
        raise ValueError(
            f"hot_vector has length {len(values)}, but num_domains is {num_domains}"
        )

    return torch.FloatTensor(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--neutral_template", required=True, choices=["white", "black", "chequered"])
    parser.add_argument("--target_path", required=True)
    parser.add_argument("--model", default="base_model", choices=["base_model", "adain_model"])
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--hot_vector", default="ones", help='Use "ones", "zeros", or a comma-separated vector like "0,1,0,0,0"')
    parser.add_argument( "--keep_resolution", action="store_true", help="Keep the original image resolution" )
    NUM_DOMAINS = 5
    INPUT_DIM = 3
    
    args = parser.parse_args()
    
    TEMPLATE_MAP = {
    "white": "assets/neutral_templates/white.jpg",
    "black": "assets/neutral_templates/black.jpg",
    "chequered": "assets/neutral_templates/chequered.jpg",
    }
    neutral_template_path = TEMPLATE_MAP[args.neutral_template]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.target_path, exist_ok=True)
    
    hot_vector = parse_hot_vector(args.hot_vector, NUM_DOMAINS)

    transform = Transform2Neutral(
        root=args.root,
        neutral_template_path=neutral_template_path,
        target_path=args.target_path,
        model=args.model,
        weight_path=args.weight_path,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        keep_resolution=args.keep_resolution,
        num_domains=NUM_DOMAINS,
        input_dim=INPUT_DIM,
        hot_vector=hot_vector,
        device=device,
    )

    transform.run()