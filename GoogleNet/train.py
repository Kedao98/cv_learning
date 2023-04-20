import json
import os
import sys
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
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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


def train_epoch(net, train_loader, loss_function, optimizer, epoch):
    net.main()
    train_bar = tqdm(train_loader, file=sys.stdout)
    running_loss = 0.0
    for step, data in enumerate(train_bar):
        images, labels = data
        outputs, aux2, aux1 = net(images.to(DEVICE))

        loss = loss_function(outputs, labels.to(DEVICE))
        loss += loss_function(aux2, labels.to(DEVICE)) * 0.3
        loss += loss_function(aux1, labels.to(DEVICE)) * 0.3

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


def train(train_dataset, val_dataset):
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
    from GoogleNet.model import GoogleNet as model

    net, model_save_path, _ = model.initialize_model_for_learning()
    net = net.to(DEVICE)
    if os.path.exists(model_save_path):
        net.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print(f'load model {model_save_path} finished')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)

    train(train_dataset="dataset/flower_data/train", val_dataset="dataset/flower_data/val")
