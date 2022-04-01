import h5py
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
from tqdm import tqdm
import data
import utils.utils as utils
import json
import resnet_master.resnet as caffe_resnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)

        def save_output(module, input, output):
            self.buffer = output

        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer


def create_coco_loader(image_size, central_fraction, batch_size, num_workers, *paths):
    transform = utils.get_transform(image_size, central_fraction)
    datasets = [data.CocoImages(path, transform=transform) for path in paths]
    dataset = data.Composite(*datasets)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return data_loader


def main(config):
    cudnn.benchmark = True
    # device = torch.device("cpu")
    # torch.device("cuda")
    net = Net().cuda()
    net.eval()
    print(config.keys())
    loader = create_coco_loader(
        config.get("image_size"),
        config.get("central_fraction"),
        config.get("preprocess_batch_size"),
        config.get("data_workers"),
        config.get("image_train"),
        config.get("image_val"),
    )
    features_shape = (
        len(loader.dataset),
        config.get("output_features"),
        config.get("output_size"),
        config.get("output_size"),
    )
    print("cuda memory ", torch.cuda.memory_allocated())
    net.cuda()
    print("here")
    with h5py.File(config.get("preprocessed_path"), "w", libver="latest") as fd:
        torch.cuda.empty_cache()
        features = fd.create_dataset("features", shape=features_shape, dtype="float16")
        coco_ids = fd.create_dataset("ids", shape=(len(loader.dataset),), dtype="int32")
        print(torch.cuda.memory_allocated())
        i = j = 0
        for ids, imgs in tqdm(loader):
            # imgs = Variable(imgs, volatile=True)
            # imgs = Variable(imgs.to(device=device), volatile=True)
            imgs = Variable(imgs.cuda(), volatile=True)

            out = net(imgs)

            j = i + imgs.size(0)
            features[i:j, :, :] = out.data.cpu().numpy().astype("float16")
            coco_ids[i:j] = ids.numpy().astype("int32")
            i = j


if __name__ == "__main__":
    with open("config.json", "r") as conf:
        config = json.loads(conf.read())
    main(config)
