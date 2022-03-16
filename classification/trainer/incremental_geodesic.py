import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geodesic import GFK

gfk = GFK()

cur_features = []
ref_features = []


def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]
    ref_features = F.adaptive_avg_pool2d(ref_features, 2).view(ref_features.size(0), -1)


def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]
    cur_features = F.adaptive_avg_pool2d(cur_features, 2).view(cur_features.size(0), -1)


def incremental_train_and_eval_LF(epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            fix_bn=False, weight_per_class=None, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        handle_ref_features = ref_model.avgpool.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.avgpool.register_forward_hook(get_cur_features)
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            outputs = tg_model(inputs)
            if iteration == start_iteration:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)  # the old features are captured from the above function (get_ref_features)
                loss1 = gfk.fit(ref_features.detach(), cur_features) * lamda  # Distillation loss
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                loss = loss1 + loss2
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('[Geodesic] Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader),
                                                                                train_loss/(batch_idx+1),
                                                                                100.*correct/total))

    if iteration > start_iteration:
        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
    return tg_model
