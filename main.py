# This is a sample Python script.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib
import keras.utils.data_utils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import itertools
import torchvision
import torchvision.transforms as transforms
import modelClasses

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dim_z = 1024
EPSILON = 1e-6
data_dir = '/Users/sumeet95/Downloads/faceDataset'
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'
batch_size = 64
lr = 0.0002
z_lr = 0.005
J1_hat = 1.00

# Image Transformation
data_transforms = {
    'train': transforms.Compose([

        transforms.ToTensor(),

    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),

    ]),
}
# Reading Dataset
image_datasets = {
    'train': ImageFolder(root=train_dir, transform=data_transforms['train']),
    'test': ImageFolder(root=test_dir, transform=data_transforms['test'])
}
total_test_batch = len(image_datasets['test']) // batch_size
# Loading Dataset
data_loaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size)
}
encoder, decoder, classifier, discriminator = modelClasses.build_CelebA_Model()


def main():
    stage1_params = itertools.chain(encoder.parameters(), decoder.parameters(), classifier.parameters())
    stage2_params = discriminator.parameters()
    optimizerStage1 = optim.Adam(stage1_params, 0.0002)
    optimizerStage2 = optim.Adam(stage2_params, 0.0002)

    for epoch in range(1):
        for train_features, train_labels in iter(data_loaders['train']):
            if len(test_features) != 64:
                break
            optimizerStage1.zero_grad()
            X, y = train_features.to(device), train_labels.to(device)
            z_mu, z_sigma = encoder(X)
            z = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
            X_hat = decoder(z)
            y_hat = classifier(z)
            classify_loss = nn.CrossEntropyLoss()(y_hat, y)
            recon_loss = nn.MSELoss(reduction='sum')(X_hat, X) / batch_size
            kl_loss = torch.mean(0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
            loss = classify_loss + recon_loss + kl_loss

            # Backward
            loss.backward()

            # Update
            optimizerStage1.step()
        with torch.no_grad():
            val_loss, val_cla, val_rec, val_kl, val_acc = 0., 0., 0., 0., 0.
            for test_features, test_labels in iter(data_loaders['test']):
                optimizerStage1.zero_grad()
                X, y = test_features.to(device), test_labels.to(device)
                z_mu, z_sigma = encoder(X)
                z = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                X_hat = decoder(z)
                y_hat = classifier(z)
                classify_loss = nn.CrossEntropyLoss()(y_hat, y)
                recon_loss = nn.MSELoss(reduction='sum')(X_hat, X) / batch_size
                kl_loss = torch.mean(
                    0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
                loss = classify_loss + recon_loss + kl_loss
                acc = torch.mean(torch.eq(torch.round(torch.sigmoid(y_hat)), y).float())

                val_loss += loss.item()
                val_cla += classify_loss.item()
                val_rec += recon_loss.item()
                val_kl += kl_loss.item()
                val_acc += acc.item()
                print('val_loss:{:.4}, val_cla:{:.4}, val_rec:{:.4}, val_kl:{:.4}, val_acc:{:.4}'.format(
                    val_loss / total_test_batch,
                    val_cla / total_test_batch,
                    val_rec / total_test_batch,
                    val_kl / total_test_batch,
                    val_acc / total_test_batch))
                if (val_loss / total_test_batch) < best_VAE_loss:
                    torch.save(encoder.state_dict(), 'out/face/encoder_best_%d.pth' % (epoch))
                    torch.save(classifier.state_dict(), 'out/face/classifier_best_%d.pth' % (epoch))
                    torch.save(decoder.state_dict(), 'out/face/generator_best_%d.pth' % (epoch))
                    best_VAE_loss = val_loss / total_test_batch

                    if best_epoch >= 0:
                        os.system('rm out/face/encoder_best_%d.pth' % (best_epoch))
                        os.system('rm out/face/classifier_best_%d.pth' % (best_epoch))
                        os.system('rm out/face/generator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(encoder.state_dict(), 'out/face/encoder_%d.pth' % (epoch))
                torch.save(classifier.state_dict(), 'out/face/classifier_%d.pth' % (epoch))
                torch.save(decoder.state_dict(), 'out/face/generator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/face/encoder_%d.pth' % (epoch - 10))
                    os.system('rm out/face/classifier_%d.pth' % (epoch - 10))
                    os.system('rm out/face/generator_%d.pth' % (epoch - 10))

    encoder.load_state_dict(torch.load('out/face/encoder_best_%d.pth' % (best_epoch)))
    best_epoch = -1

    for p in encoder.parameters():
        p.requires_grad = False
    for epoch in range(1):
        for train_features, train_labels in iter(data_loaders['train']):
            if len(test_features) != 64:
                break
            optimizerStage2.zero_grad()
            X, y = train_features.to(device), train_labels.to(device)
            optimizerStage2.zero_grad()

            # Forward
            z_mu, z_sigma = encoder(X)
            z_real = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
            z_fake = torch.randn(batch_size, dim_z).to(device)

            D_real = discriminator(z_real)
            D_fake = discriminator(z_fake)

            # Loss
            loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))

            # Backward
            loss.backward()

            # Update
            optimizerStage2.step()
        with torch.no_grad():
            # validate after each epoch
            val_loss_dis, acc_dis_true, acc_dis_fake = 0., 0., 0.

            for test_features, test_labels in iter(data_loaders['test']):
                if len(test_features) != 64:
                    break
                X, y = test_features.to(device), test_features.to(device)

                z_mu, z_sigma = encoder(X)
                z_real = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                z_fake = torch.randn(batch_size, dim_z).to(device)

                D_real = discriminator(z_real)
                D_fake = discriminator(z_fake)

                loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))

                val_loss_dis += loss.item()
                acc_dis_true += torch.mean(torch.ge(D_real, 0.5).float()).item()
                acc_dis_fake += torch.mean(torch.lt(D_fake, 0.5).float()).item()

            print('val_loss_dis:{:.4}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(val_loss_dis / total_test_batch,
                                                                                      acc_dis_true / total_test_batch,
                                                                                      acc_dis_fake / total_test_batch))
            if (val_loss_dis / total_test_batch) < best_DIS_loss:
                torch.save(discriminator.state_dict(), 'out/face/discriminator_best_%d.pth' % (epoch))
                best_DIS_loss = val_loss_dis / total_test_batch

                if best_epoch >= 0:
                    os.system('rm out/face/discriminator_best_%d.pth' % (best_epoch))

                best_epoch = epoch

        if epoch % 10 == 0:
            torch.save(discriminator.state_dict(), 'out/face/discriminator_%d.pth' % (epoch))

            if epoch >= 10:
                os.system('rm out/face/discriminator_%d.pth' % (epoch - 10))

    """print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img[0])
    plt.show()
    print(f"Label: {label}")"""


def generate():
    test_img_index = int(sys.argv[2])
    x_test, y_test = data_loaders['test']

    # choose test image and its target label
    test_img = x_test[test_img_index, ...]
    test_label = y_test[test_img_index, ...]
    print('original_label is :', test_label.tolist())
    target_label = 1 - test_label
    print("target_label is ", target_label.tolist())

    img = test_img.transpose((1, 2, 0))
    plt.imsave('attack/face/test.png', img)

    encoder.load_state_dict(torch.load('out/face/encoder.pth'))
    decoder.load_state_dict(torch.load('out/face/generator.pth'))
    classifier.load_state_dict(torch.load('out/face/classifier.pth'))
    discriminator.load_state_dict(torch.load('out/face/discriminator.pth'))

    for p in encoder.parameters():
        p.requires_grad = False

    for p in decoder.parameters():
        p.requires_grad = False

    for p in classifier.parameters():
        p.requires_grad = False

    for p in discriminator.parameters():
        p.requires_grad = False

    X, y = test_img.to(device), test_label.to(device)
    y_hat = torch.from_numpy(np.expand_dims(target_label, 0)).float().to(device)

    X_embedding = facenet(F.interpolate(X, size=(160, 160), mode='bilinear', align_corners=True))

    z, _ = encoder(X)
    z = Variable(z, requires_grad=True).to(device)

    z_solver = optim.Adam([z], lr=z_lr)

    k = 0
    iter_num = 20000

    for it in range(iter_num + 1):
        z_solver.zero_grad()

        # Forward
        y2 = classifier(z)
        D = discriminator(z)
        X_hat = decoder(z)

        X_hat_embedding = facenet(F.interpolate(X_hat, size=(160, 160), mode='bilinear', align_corners=True))

        # loss
        J1 = distance(X_embedding, X_hat_embedding)
        J2 = nn.CrossEntropyLoss()(y2, y_hat)
        J_IT = J2 + 0.01 * torch.mean(1 - torch.sigmoid(D)) + 0.0001 * torch.mean(torch.norm(z, dim=1))
        J_SA = J_IT + k * J1
        k += z_lr * (0.001 * J1.item() - J2.item() + max(J1.item() - J1_hat, 0))
        k = max(0, min(k, 0.005))

        if (it % 1000 == 0):
            print('iter-%d: J_SA: %.4f, J_IT: %.4f, J1: %.4f' % (it, J_SA.item(), J_IT.item(), J1.item()))
            img = torch.squeeze(X_hat).permute(1, 2, 0).data.cpu().numpy()
            plt.imsave('attack/CelebA/iter-%d.png' % (it), img)

        # Backward
        J_SA.backward()

        # Update
        z_solver.step()


def distance(x1, x2):
    return torch.mean(torch.norm(x1 - x2, dim=1))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if sys.argv[0] == "train":
        main()
    else:
        generate()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
