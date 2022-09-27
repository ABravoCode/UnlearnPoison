import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm

from models.ResNet import ResNet18
import toolbox
from util import AverageMeter

import random
import matplotlib.pyplot as plt
import matplotlib

CLASS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def find_similar_img(target, k=50):
    trainDataLoader = torch.utils.data.DataLoader(dataset=clean_train_dataset, batch_size=1)
    processBar = tqdm(trainDataLoader, unit='step')
    sims = {}
    for index, (trainImgs, labels) in enumerate(processBar):
        trainImgs = trainImgs.to(device)
        cos_sim = torch.nn.functional.cosine_similarity(target.flatten(), trainImgs.flatten(), dim=0)
        sims[index] = cos_sim
    sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    # print(sims[:10])  # (9881, tensor(0.9158))
    # print(sims[0][0])  # 9881
    
    sim_group = []
    for i in range(k):
        sim_group.append(sims[i][0])

    return sim_group

def linear_interpolation(target_img, sim_img_group, alpha=0.1):
    ip_img = []
    for img in sim_img_group:
        interpolation = alpha * target_img + (1-alpha) * img
        ip_img.append(interpolation)
    return ip_img

def generate_noise(base_model, criterion, optimizer, interpolation_imgs, intended_label, MAX_ITERATION=10):
    noise = torch.zeros([len(interpolation_imgs), 3, 32, 32])
    data_iter = iter(interpolation_imgs)
    condition = True
    train_idx = 0

    while condition:
        base_model.train()
        for param in base_model.parameters():
            param.requires_grad = True
        for j in range(0, MAX_ITERATION):
            try:
                images = next(data_iter)
            except:
                train_idx = 0
                data_iter = iter(clean_train_loader)
                images = next(data_iter)
            
            for i, _ in enumerate(images):
                images[i] += noise[train_idx]
                train_idx += 1
            images, intended_labels = images.cuda(), intended_labels.cuda()
            base_model.zero_grad()
            optimizer.zero_grad()
            logits = base_model(images)
            loss = criterion(logits, intended_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
            optimizer.step()
        
        idx = 0
        for param in base_model.parameters():
            param.requires_grad = False
        for i, (images, labels) in tqdm(enumerate(interpolation_imgs), total=len(interpolation_imgs)):
            batch_start_idx, batch_noise = idx, []
            for i, _ in enumerate(images):
                batch_noise.append(noise[idx])
                idx += 1
            batch_noise = torch.stack(batch_noise).cuda()
            
            base_model.eval()
            images, labels = images.cuda(), labels.cuda()
            perturb_img, eta = noise_generator.min_min_attack(images, labels, base_model, optimizer, criterion, 
                                                            random_noise=batch_noise)
            for i, delta in enumerate(eta):
                noise[batch_start_idx+i] = delta.clone().detach().cpu()
            
        eval_idx, total, correct = 0, 0, 0
        for i, (images, labels) in enumerate(clean_train_loader):
            for i, _ in enumerate(images):
                images[i] += noise[eval_idx-1]
                eval_idx += 1
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = base_model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        print('Accuracy %.6f' % (acc*100))
        if acc > 0.95:
            condition=False  
        return images, noise

def create_poison_dataset(new_imgs):
    poison_dataset = new_imgs
    poison_dataset.data = poison_dataset.data.astype(np.float32)
    for i in range(len(poison_dataset)):
        poison_dataset.data[i] = np.clip(poison_dataset.data[i], a_min=0, a_max=255)
    poison_dataset.data = poison_dataset.data.astype(np.uint8)
    return poison_dataset

def train_new_poisoned_model(poison_dataset, MAX_EPOCH=20, BATCH_SIZE=128):
    model = ResNet18()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)

    unlearnable_loader = DataLoader(poison_dataset, BATCH_SIZE=BATCH_SIZE,
                                    shuffle=True, pin_memory=True,
                                    drop_last=False, num_workers=12)
    clean_train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
    clean_loader = DataLoader(clean_train_dataset, BATCH_SIZE=BATCH_SIZE,
                                    shuffle=True, pin_memory=True,
                                    drop_last=False, num_workers=12)
    clean_test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)
    clean_test_loader = DataLoader(dataset=clean_test_dataset, BATCH_SIZE=BATCH_SIZE,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    
    for epoch in range(MAX_EPOCH):
        model.train()
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        pbar = tqdm([unlearnable_loader, clean_loader], total=len(unlearnable_loader, clean_loader))
        for images, labels in pbar:
            images, labels = images.cuda(), labels.cuda()
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            _, predicted = torch.max(logits.data, 1)
            acc = (predicted == labels).sum().item()/labels.size(0)
            acc_meter.update(acc)
            loss_meter.update(loss.item())
            pbar.set_description("Acc %.4f Loss: %.4f" % (acc_meter.avg*100, loss_meter.avg))
        scheduler.step()
        # Eval
        model.eval()
        correct, total = 0, 0
        for i, (images, labels) in enumerate(clean_test_loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        tqdm.write('Clean Accuracy %.4f\n' % (acc*100))
        # Test target class
        target = torch.tensor(clean_test_dataset.data[8745]).to(device)
        with torch.no_grad():
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            # print(CLASS[int(predicted)])
    return predicted


if __name__ == '__main__':
    root = './datasets/'
    BATCH_SIZE = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_POISON = 50

    train_transform = [
        transforms.ToTensor()
    ]
    test_transform = [
        transforms.ToTensor()
    ]
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    clean_train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
    clean_test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=test_transform)

    clean_train_loader = DataLoader(dataset=clean_train_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    clean_test_loader = DataLoader(dataset=clean_test_dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, pin_memory=True,
                                    drop_last=False, num_workers=12)
    
    target = torch.tensor(clean_test_dataset.data[8745]).to(device)
    label = 5

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    base_model = ResNet18()
    base_model = base_model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=base_model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)

    noise_generator = toolbox.PerturbationTool(epsilon=0.03137254901960784, num_steps=20, step_size=0.0031372549019607846)

    similar_imgs = find_similar_img(target, NUM_POISON)
    interpolation_imgs = linear_interpolation(target, similar_imgs, alpha=0.5)
    perturb_img, noise = generate_noise(base_model, criterion, optimizer, interpolation_imgs, intended_label=3, MAX_ITERATION=10)
    poison_dataset = create_poison_dataset(perturb_img)

    def imshow(img):
        fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    def get_pairs_of_imgs(idx):
        clean_img = clean_train_dataset.data[idx]
        unlearnable_img = poison_dataset.data[idx]
        clean_img = torchvision.transforms.functional.to_tensor(clean_img)
        unlearnable_img = torchvision.transforms.functional.to_tensor(unlearnable_img)

        x = noise[idx]
        x_min = torch.min(x)
        x_max = torch.max(x)
        noise_norm = (x - x_min) / (x_max - x_min)
        noise_norm = torch.clamp(noise_norm, 0, 1)
        return [clean_img, noise_norm, unlearnable_img]
        
    selected_idx = [random.randint(0, NUM_POISON) for _ in range(3)]
    img_grid = []
    for idx in selected_idx:
        img_grid += get_pairs_of_imgs(idx)
    
    imshow(torchvision.utils.make_grid(torch.stack(img_grid), nrow=3, pad_value=255))

    predict = train_new_poisoned_model(poison_dataset, MAX_EPOCH=20, BATCH_SIZE=128)
    print('Target 8745[dog] is classified as type {}'.format(CLASS[int(predict)]))