import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import joblib

from data.data_loader import MyDataset, loader
from models.gbdt_dnn import GbdtDnn
from models.focal_loss import FocalLoss


torch.manual_seed(555)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def main(args):
    model = GbdtDnn(args.input_size, args.n_class).to(device)

    # setup optimizer
    # optimizer = optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=.01)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.beta1, 0.99))

    gbdt_model = joblib.load(args.gbdt_model_path)

    train_data, train_labels = \
        get_features_and_labels(args.train_data_file_path, args.target)
    test_data, test_labels = get_features_and_labels(args.test_data_file_path,
                                                     args.target)

    train_dataset = MyDataset(train_data, train_labels, gbdt_model)
    test_dataset = MyDataset(test_data, test_labels, gbdt_model)
    train_loader = loader(train_dataset, args.batch_size)
    test_loader = loader(test_dataset, 1, shuffle=False)

    train(args, model, optimizer, train_loader)
    test(args, model, test_loader)


def get_features_and_labels(file_path, target):
    data = pd.read_csv(file_path)
    y = data[target].values
    X = data.drop(target, axis=1).values
    return X, y


def l1_loss(model, reg=1e-4):
    loss = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            loss += reg * torch.sum(torch.abs(param))
    return loss


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=70):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    for param_group in optimizer.param_groups:
        decaied = param_group['lr'] * 0.5
        param_group['lr'] = init_lr if decaied <= 2e-7 else decaied

    print("decay lr")
    return optimizer


def train(args, model, optimizer, data_loader):
    model.train()
    min_loss = 1e6
    no_implove_cnt = 0
    criteria = FocalLoss(args.n_class, ignore_label=-1)
    for epoch in range(args.epochs):
        for i, (l_data, l_target) in enumerate(data_loader):
            l_data = l_data.to(device)
            l_target = l_target.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            output = model(l_data)
            targets = l_target.view(-1)
            loss = criteria(output, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

            min_loss = loss.item() if min_loss > loss.item() else min_loss
            no_implove_cnt = no_implove_cnt + 1 if loss.item() > min_loss else 0
            if no_implove_cnt == 250:
                exp_lr_scheduler(optimizer, epoch, init_lr=args.lr)
                no_implove_cnt = 0

        # do checkpointing
        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       '{}/{}_model_ckpt.pth'.format(args.out_dir, args.target))
    torch.save(model.state_dict(),
               '{}/{}_model_ckpt.pth'.format(args.out_dir, args.target))


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i_batch, (l_data, l_target) in enumerate(data_loader):
            l_data = l_data.to(device)
            l_target = l_target.to(device)

            output = model(l_data)
            targets = l_target.view(-1)
            test_loss += F.cross_entropy(output, targets, ignore_index=-1).item()

            pred = output.argmax(1)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    total_len = len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, total_len, 100. * correct / total_len))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-class', type=int, default=4, help='number of class')
    parser.add_argument('--input-size', type=int, default=200)
    parser.add_argument('--target', default='period')
    parser.add_argument('--gbdt-model-path', default="./results/trained_xgb_model.model")
    parser.add_argument('--train-data-file-path', default='./data/prepreprocessed_train.csv', help='path to train data file pointer')
    parser.add_argument('--test-data-file-path', default='./data/prepreprocessed_test.csv', help='path to test data file pointer')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.9')
    parser.add_argument('--out-dir', default='./results', help='folder to output data and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
