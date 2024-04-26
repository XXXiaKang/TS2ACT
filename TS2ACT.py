import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import argparse
from utils import load_image, load_text
from torch import optim
from torch.utils.data import DataLoader
from dataset.UCI import *
from dataset.PAMAP2 import *
from dataset.HHAR import *
from dataset.MotionSense import *
from models.net import *
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
device = torch.device('cuda')
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/home/xiakang/data',
                        help="path to Dataset")
    parser.add_argument("--checkpoint_root", type=str, default='output',
                        help="path to checkpoint")
    dataset_name = 'HHAR'
    parser.add_argument("--dataset", type=str, default= dataset_name,
                        choices=['UCI','UCI2','PAMAP2','MotionSense'], help='Name of dataset')

    parser.add_argument('--shot', type=str,
                        default='10-shot')
    parser.add_argument('--model', type=str, help='the name of net',
                        default='TimeT')
    parser.add_argument('--test', type=int, help = '', default = 1)
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=10000)

    train_type = 'contrastive_image'
    test_type = 'contrastive_text' 
    parser.add_argument("--train_type", type=str, default= train_type,
        choices=['classify','contrastive','contrastive_text','contrastive_image'],)
    parser.add_argument("--test_type", type=str, default= test_type,
        choices=['classify','contrastive_text'])
    parser.add_argument('--iterations',type=int,help='number of episodes per epoch, default=100',default=100)
    if dataset_name in ['UCI']: 
        parser.add_argument("--num_feature", type=int, default=9)
        parser.add_argument("--num_class", type=int, default=9)
        parser.add_argument("--clip_len", type=int, default=500)
    elif dataset_name in ['PAMAP2']: 
        parser.add_argument("--num_feature", type=int, default=36)
        parser.add_argument("--num_class", type=int, default=12)
        parser.add_argument("--clip_len", type=int, default=500)
    elif dataset_name in ['MotionSense']: 
        parser.add_argument("--num_feature", type=int, default=12)
        parser.add_argument("--num_class", type=int, default=6)
        parser.add_argument("--clip_len", type=int, default=500)                                         
    elif dataset_name in ['HHAR']: 
        parser.add_argument("--num_feature", type=int, default=12)
        parser.add_argument("--num_class", type=int, default=6)
        parser.add_argument("--clip_len", type=int, default=500)   
    return parser

def train_model(opt, model, optimizer, scheduler, labeled_dst , unlabeled_tra=None, unlabeled_val=None,best_acc = 0.0):
    Epochs = opt.epochs
    labeled_dataloader = DataLoader( labeled_dst, batch_size = opt.batch_size, shuffle=True)
    unlabeled_dataloader = DataLoader( unlabeled_tra, batch_size = opt.val_batch_size, shuffle=True)
    if unlabeled_val is not None:
        val_dataloader = DataLoader( unlabeled_val, batch_size = opt.val_batch_size)
    else:val_dataloader = None
    clip_text = load_text(opt)
    clip_image = load_image(opt) 
    for epoch in range(Epochs):
        print("lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        val_sum = val_count = labeled_loss = labeled_corrects = count = 0.0
        model.train()
        save_train_fea_X = []
        save_train_fea_y = []
        labeled_iter = iter(labeled_dataloader)
        unlabeled_iter = iter(unlabeled_dataloader)
        for batch_idx in range(100):
            try:
                inputs_x, y = labeled_iter.next()
            except:
                labeled_iter = iter(labeled_dataloader)
                inputs_x, y = labeled_iter.next()
            try:
                inputs_u_w, inputs_u_s, y_u = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_dataloader)
                inputs_u_w, inputs_u_s, y_u = unlabeled_iter.next()
            optimizer.zero_grad()
            inputs_x = inputs_x.cuda().float()
            y = y.cuda().long().view(-1)
            inputs_u_w = inputs_u_w.cuda().float()
            inputs_u_s = inputs_u_s.cuda().float()
            y_u = y_u.cuda().view(-1)
            features,features2class,mlp_features = model(inputs_x)
            features = features / features.norm(dim=-1, keepdim=True)
            if opt.train_type == 'contrastive':
                criterion = torch.nn.CrossEntropyLoss().cuda()
                A_features, A_features2class, A_mlp_features = model(inputs_u_w) 
                B_features, B_features2class, B_mlp_features = model(inputs_u_s)       
                A_features = A_features / A_features.norm(dim=-1, keepdim=True)
                B_features = B_features / B_features.norm(dim=-1, keepdim=True)
                dists = A_features @ B_features.T  / 0.07
                target = torch.arange(0, dists.shape[0]).view(-1).cuda()
                pre = torch.argmax(dists, dim = 1)
                confuse = (pre == target)
                val_sum += confuse.float().sum()
                val_count += confuse.shape[0]
                loss = criterion(dists, target)
                acc = (torch.argmax(dists, dim = 1) == target).float().mean()

            if opt.train_type == 'contrastive_image':
                if clip_image is None:
                    print("clip_image is None!")
                    exit()
                code = clip_image[torch.arange(0, opt.num_class).cuda(),torch.randint(0, 14, (opt.num_class,)).cuda()]
                dists = features @ code.t() / 0.07
                criterion = torch.nn.CrossEntropyLoss().cuda()
                loss = criterion(dists, y)
                acc = (torch.argmax(dists, dim = 1) == y).float().mean()
                A_features, A_features2class, A_mlp_features = model(inputs_u_w) 
                B_features, B_features2class, B_mlp_features = model(inputs_u_s)       
                A_features = A_features / A_features.norm(dim=-1, keepdim=True)
                B_features = B_features / B_features.norm(dim=-1, keepdim=True)

                dists1 = A_features @ code.t() / 0.07
                pre1 = torch.argmax(dists1, dim = 1)
                confuse = (pre1 == y_u)
                val_sum += confuse.float().sum()
                val_count += confuse.shape[0]

                dists2 = B_features @ code.t() / 0.07
                pre2 = torch.argmax(dists2, dim = 1)
                loss += criterion(dists2, pre1) * 0.01

            if opt.train_type == 'contrastive_text':
                dists = features @ clip_text.t() / 0.07
                criterion = torch.nn.CrossEntropyLoss().cuda()
                loss = criterion(dists, y)
                acc = (torch.argmax(dists, dim = 1) == y).float().mean()


                A_features, A_features2class, A_mlp_features = model(inputs_u_w) 
                B_features, B_features2class, B_mlp_features = model(inputs_u_s)       
                A_features = A_features / A_features.norm(dim=-1, keepdim=True)
                B_features = B_features / B_features.norm(dim=-1, keepdim=True)

                dists1 = A_features @ clip_text.t() / 0.07
                pre1 = torch.argmax(dists1, dim = 1)
                confuse = (pre1 == y_u)
                val_sum += confuse.float().sum()
                val_count += confuse.shape[0]

                dists2 = B_features @ clip_text.t() / 0.07
                pre2 = torch.argmax(dists2, dim = 1)
                loss += criterion(dists2, pre1) * 0.01


            if opt.train_type == 'classify':
                criterion = torch.nn.CrossEntropyLoss().cuda()
                loss = criterion(features2class, y)
                # acc = (torch.argmax(dists, dim = 1) == y).float().mean()

                A_features, A_features2class, A_mlp_features = model(inputs_u_w) 
                B_features, B_features2class, B_mlp_features = model(inputs_u_s)       

                pre1 = torch.argmax(A_features2class, dim = 1)
                confuse = (pre1 == y_u)
                val_sum += confuse.float().sum()
                val_count += confuse.shape[0]

                loss += criterion(B_features2class, pre1) * 0.01
                acc = (torch.argmax(B_features2class, dim = 1) == pre1).float().mean()

            loss.backward()
            optimizer.step()
            count += 1.0
            labeled_loss += loss.item() 
            labeled_corrects += acc.item()


        epoch_loss = labeled_loss / count
        epoch_acc = labeled_corrects / count

        if scheduler != None:
            scheduler.step() 
        if opt.test==0:
            epoch_loss = labeled_loss / count
            epoch_acc = labeled_corrects / count
            val_acc = val_sum.item() / val_count if val_count > 0.0 else 0.0
            postfix = ' (Best)' if val_acc >= best_acc else ' (Best: {})'.format(best_acc)
            print("Label:{} Acc: {}{} ".format(epoch_acc, val_acc, postfix))
            if val_acc >= best_acc:
                best_acc = val_acc
                best_model_path = save_path(opt, 'best')
                torch.save({
                    'epoch': epoch + 1,
                    'acc':val_acc,
                    'state_dict': model.state_dict(),
                }, best_model_path)
            print("Save model at {}".format(best_model_path))
            continue
        val_acc = val_sum.item() / val_count if val_count > 0.0 else 0.0
        print("[train] Label:{} Unlabel:{} ".format(epoch_acc, val_acc))
        if (epoch+1) % 10 != 0:
            continue
        model.eval()
        labeled_corrects = count = 0.0
        if val_dataloader is not None: unlabeled_iter = iter(val_dataloader)
        else: unlabeled_iter = iter(unlabeled_dataloader)
        running_corrects = 0.0
        y_true = []
        y_pred = []
        for X,_,y in unlabeled_iter:
            X = X.cuda().float()
            y = y.cuda().view(-1)
            with torch.no_grad():
                features, features2class, mlp_features = model(X)           
                features = features / features.norm(dim=-1, keepdim=True)
            if opt.test_type == 'contrastive_text':
                x = features @ clip_text.t()
                acc = (torch.argmax(x, dim = 1) == y).float().mean()
                y_true+= y.tolist()
                y_pred+= torch.argmax(x, dim = 1).tolist()
            if opt.test_type == 'classify':
                acc = (torch.argmax(features2class, dim = 1) == y).float().mean()
                y_true+= y.tolist()
                y_pred+= torch.argmax(features2class, dim = 1).tolist()
            count += 1.0
            running_corrects += acc.item()
        val_acc = running_corrects / count
        postfix = ' (Best)' if val_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print("[{}] f1:{} recall:{} Acc: {}{} ".format('val',
                                                       f1_score(y_true, y_pred, average='macro'),
                                                       recall_score(y_true, y_pred, average='macro'),
                                                       val_acc,postfix))
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model_path = save_path(opt, 'best')
            torch.save({
                'epoch': epoch + 1,
                'acc':val_acc,
                'f1':f1_score(y_true, y_pred, average='macro'),
                'recall':recall_score(y_true, y_pred, average='macro'),
                'state_dict': model.state_dict(),
            }, best_model_path)
            print("Save model at {}".format(best_model_path))

def init_dataloader(opt):
    if opt.dataset == 'MotionSense':
        labeled_dst ,unlabeled_tra = MotionSense_TS2ACT_aug(clip_len=opt.clip_len, name = opt.shot)
    if opt.dataset == 'PAMAP2':
        labeled_dst ,unlabeled_tra = PAMAP_TS2ACT_aug(clip_len=opt.clip_len,name = opt.shot)
    if opt.dataset == 'HHAR':
        labeled_dst ,unlabeled_tra = HHAR_TS2ACT_aug(clip_len=opt.clip_len,name = opt.shot)
    if opt.dataset == 'UCI':
        labeled_dst ,unlabeled_tra = UCI_TS2ACT_aug(clip_len=opt.clip_len,name = opt.shot)
    c = None
    return labeled_dst ,unlabeled_tra, c


def save_path(opt, split):
    return os.path.join(opt.checkpoint_root, 
                        opt.model + '_' +
                        opt.shot + '_' +
                        opt.train_type + '_' + 
                        opt.dataset + '_' + 
                        split +'.pth')
def main(opt):
    model = TimeTransformer(opt.clip_len,opt.num_feature, 512 ,layers= 6,heads=16, output_dim=opt.num_class, dropout=0.0)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.008, momentum = 0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    best_acc = 0.0
    labeled_dst , unlabeled_tra, unlabeled_test = init_dataloader(opt)

    if opt.conti == 1:
        last_model_path = save_path(opt, 'last')
        best_model_path = save_path(opt, 'best')
        checkpoint = torch.load(best_model_path)
        best_acc = checkpoint['acc']
        print("Initializing weights from: {}...".format(last_model_path))
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['state_dict'])
    train_model(opt, model, optimizer, scheduler, labeled_dst ,unlabeled_tra, unlabeled_test,best_acc=best_acc)


if __name__ == "__main__":
    opts = get_argparser().parse_args()
    main(opts)