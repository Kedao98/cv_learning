import json
import os
import sys
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim

# device info
DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f"using {DEVICE} device.")

# dataloader config
DATA_TRANSFORM = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
# train config
EPOCHS = 30
BATCH_SIZE = 16
NUM_WORKERS = 1
print(f'Using {NUM_WORKERS} dataloader workers every process')


def load_dataset(train_dataset, val_dataset):
    # load dataset & dataloader
    assert os.path.exists(train_dataset) and os.path.exists(val_dataset), "dataset path does not exist."

    train_dataset = datasets.ImageFolder(root=train_dataset, transform=DATA_TRANSFORM['train'])
    val_dataset = datasets.ImageFolder(root=val_dataset, transform=DATA_TRANSFORM['val'])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=NUM_WORKERS)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=False,
                                                  num_workers=NUM_WORKERS)

    # save label
    with open(f"{os.path.dirname(model_save_path)}/class_indices.json", "w") as f:
        class_dict = dict((val, key) for key, val in train_dataset.class_to_idx.items())
        f.write(json.dumps(class_dict, indent=4))
    print(f"using {len(train_dataset)} images for training, {len(val_dataset)} images for validation.")
    return train_dataset, val_dataset, train_loader, validate_loader


def load_model_weights(net, model_save_path, mode=''):
    # load your own pth
    if os.path.exists(model_save_path):
        net.load_state_dict(torch.load(model_save_path, map_location='cpu'))
        print(f'load model {model_save_path} finished')

    # load pre-trained pth
    elif glob(f"{os.path.dirname(model_save_path)}/*.pth"):
        model_save_path = glob(f"{os.path.dirname(model_save_path)}/*.pth")[0]
        weights = torch.load(model_save_path, map_location='cpu')
        weights = {k: v for k, v in weights.items() if net.state_dict()[k].numel() == v.numel()}

        net.load_state_dict(weights, strict=False)
        print(f'load model {model_save_path} finished')

        if mode == 'freeze':
            for param in net.features.parameters():
                param.requires_grad = False

    net = net.to(DEVICE)
    params = [p for p in net.parameters() if p.requires_grad]
    return net, params


def train_epoch(net, train_loader, loss_function, optimizer, epoch):
    net.train()
    train_bar = tqdm(train_loader, file=sys.stdout)
    running_loss = 0.0
    for step, data in enumerate(train_bar):
        images, labels = data
        outputs = net(images.to(DEVICE))
        loss = loss_function(outputs, labels.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = f"train epoch[{epoch + 1}/{EPOCHS}] loss:{loss:.3f}"
    return running_loss


def validate_epoch(net, validate_loader):
    net.eval()
    acc_cnt = 0.0
    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(DEVICE))
            predict_y = torch.max(outputs, dim=1)[1]
            acc_cnt += torch.eq(predict_y, val_labels.to(DEVICE)).sum().item()
    return acc_cnt


def main(net, loss_function, optimizer, train_dataset, val_dataset):
    train_dataset, val_dataset, train_loader, validate_loader = load_dataset(train_dataset, val_dataset)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        # train
        running_loss = train_epoch(net, train_loader, loss_function, optimizer, epoch)

        acc_cnt = validate_epoch(net, validate_loader)
        val_accurate = acc_cnt / len(val_dataset)
        print(f'[epoch {epoch + 1}] '
              f'train_loss: {running_loss / len(train_loader):.3f} '
              f'val_accuracy: {val_accurate:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), model_save_path)


if __name__ == '__main__':
    from ShuffleNet.model import ShuffleNetV2 as model

    net, model_save_path, _ = model.initialize_model_for_learning()
    net, net_params = load_model_weights(net, model_save_path, mode='unfreeze')  # freeze

    main(net, loss_function=nn.CrossEntropyLoss(), optimizer=optim.Adam(net_params, lr=1e-5),
         train_dataset="dataset/flower_data/train", val_dataset="dataset/flower_data/val")
