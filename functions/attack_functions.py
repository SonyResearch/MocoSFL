# This function means performing training of the attacker's inversion model, is used in MIA_attack function.
    
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
from torchvision.utils import save_image
from utils import setup_logger, AverageMeter
from functions import pytorch_ssim
from shutil import rmtree
import os
from PIL import Image
from glob import glob
import datasets
from datasets import denormalize
import numpy as np
import torchvision.transforms as transforms
from functions.pytorch_ssim import SSIM

NUM_CHANNEL_GROUP = 4

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

class MIA_simulator():
    def __init__(self, model, args, MIA_arch = "custom") -> None:
        self.model = model
        self.local_AE_list = []
        self.gan_params = []
        self.gan_AE_activation = "sigmoid"
        self.data_size = args.data_size
        self.client_sample_ratio = args.client_sample_ratio
        self.dataset = args.dataset
        self.ssim_threshold = args.ressfl_target_ssim
        self.num_client = args.num_client
        self.alpha = args.ressfl_alpha
        self.gan_multi_step = 1
        self.gan_decay = 0.2

        self.MIA_arch = MIA_arch
        if args.batch_size < 20:
            GN_option = True
            WS_option = True
        else:
            GN_option = False
            WS_option = True
        # self.gan_loss_type = "SSIM"

        input_nc, input_dim = self.model.get_smashed_data_size(1, self.data_size)[1:3]
        
        for i in range(self.num_client):
            if self.MIA_arch == "custom":
                self.local_AE_list.append(custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=self.data_size,
                                                    activation=self.gan_AE_activation))
            elif "conv_normN" in self.MIA_arch:
                try:
                    afterfix = self.MIA_arch.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                self.local_AE_list.append(conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=self.data_size,
                                                            activation=self.gan_AE_activation, GN_option = GN_option, WS = WS_option))
            elif "res_normN" in self.MIA_arch:
                try:
                    afterfix = self.MIA_arch.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                self.local_AE_list.append(res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=self.data_size,
                                                            activation=self.gan_AE_activation, GN_option = GN_option, WS = WS_option))
            
            else:
                raise ("No such GAN AE type.")

            self.gan_params.append(self.local_AE_list[i].parameters())
            self.local_AE_list[i].apply(init_weights)
        self.gan_optimizer_list = []
        self.gan_scheduler_list = []
        milestones = [60, 120, 160]

        if self.client_sample_ratio < 1.0:
            multiplier = 1/self.client_sample_ratio
            for i in range(len(milestones)):
                milestones[i] = int(milestones[i] * multiplier)
        for i in range(len(self.gan_params)):
            self.gan_optimizer_list.append(torch.optim.Adam(list(self.gan_params[i]), lr=1e-3))
            self.gan_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.gan_optimizer_list[i], milestones=milestones,
                                                                      gamma=self.gan_decay))  # learning rate decay

    def eval(self, client_id, act):
        self.local_AE_list[client_id].eval()
        output_image = self.local_AE_list[0](act)
        return output_image

    def train(self, client_id, act, query):
        self.local_AE_list[client_id].train()
        
        for _ in range(self.gan_multi_step):
            output_image = self.local_AE_list[client_id](act.detach()) # feed activation to AE
            query = denormalize(query, self.dataset)
            ssim_loss = SSIM()
            loss = -ssim_loss(output_image, query)

            self.gan_optimizer_list[client_id].zero_grad()
            loss.backward()

            self.gan_optimizer_list[client_id].step()

        losses = -loss.detach().cpu().numpy()
        del loss

        return losses

    def cuda(self):
        for i in range(self.num_client):
            self.local_AE_list[i].cuda()
    
    def cpu(self):
        for i in range(self.num_client):
            self.local_AE_list[i].cpu()

    def regularize_grad(self, client_id, act, query):
        act.requires_grad=True

        act.retain_grad()

        self.local_AE_list[client_id].eval()
        output_image = self.local_AE_list[client_id](act)
        query = denormalize(query, self.dataset)
        ssim_loss = SSIM()
        ssim_term = ssim_loss(output_image, query)

        if self.ssim_threshold > 0.0:
            if ssim_term > self.ssim_threshold:
                gan_loss = self.alpha * (ssim_term - self.ssim_threshold) # Let SSIM approaches 0.4 to avoid overfitting
            else:
                error = ssim_term.detach().cpu().numpy()
                return error, None # if ssim is already small, do not produce grad
        else:
            gan_loss = self.alpha * ssim_term

        error = ssim_term.detach().cpu().numpy()

        if act.grad is not None:
            act.grad.zero_()
        
        gan_loss.backward()

        gradient = act.grad.detach().clone()

        return error, gradient
    
    def scheduler_step(self):
        for i in range(self.num_client):
            self.gan_scheduler_list[i].step()
    
class MIA_attacker():
    def __init__(self, model, train_loader, args, MIA_arch = "custom") -> None:
        self.save_dir = args.output_dir
        self.num_class = args.num_class
        self.data_size = args.data_size
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.train_loader = train_loader[0]
        self.atk_data_proportion = 0.01
        model_log_file = self.save_dir + '/MIA.log'
        logger = setup_logger('attack_logger', model_log_file, level=logging.DEBUG)
        self.logger = logger
        self.model = model
        self.MIA_arch = MIA_arch
        self.gan_AE_activation = "sigmoid"
        
    # Main function to do Model Inversion attack, we support model-based ("MIA") and optimization-based ("MIA_mf")
    def MIA_attack(self, attack_option="MIA", target_client=0):
        attack_option = attack_option
        MIA_lr = 1e-3
        attack_batchsize = 32
        attack_num_epochs = 50
        
        # pass
        image_data_dir = self.save_dir + "/img"
        tensor_data_dir = self.save_dir + "/img"

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)

        create_atk_dataset = getattr(datasets, f"get_{self.dataset}_trainloader")
        val_single_loader = create_atk_dataset(batch_size=1, num_workers=4, shuffle=False, data_portion = self.atk_data_proportion)[0]

        attack_path = self.save_dir + '/MIA_temp'
        if not os.path.isdir(attack_path):
            os.makedirs(attack_path)
            os.makedirs(attack_path + "/train")
            os.makedirs(attack_path + "/test")
            os.makedirs(attack_path + "/tensorboard")
            os.makedirs(attack_path + "/sourcecode")
        
        train_output_path = "{}/train".format(attack_path)
        test_output_path = "{}/test".format(attack_path)
        tensorboard_path = "{}/tensorboard/".format(attack_path)

        model_path = "{}/model.pt".format(attack_path)
        path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                     "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

        self.logger.debug("Generating IR ...... (may take a while)")

        self.gen_ir(val_single_loader, self.model.local_list[0], image_data_dir, tensor_data_dir)
        
        input_nc, input_dim = self.model.get_smashed_data_size(1, self.data_size)[1:3]

        if self.MIA_arch == "custom":
            decoder = custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=self.data_size,
                                                activation=self.gan_AE_activation).cuda()
        elif "conv_normN" in self.MIA_arch:
            try:
                afterfix = self.MIA_arch.split("conv_normN")[1]
                N = int(afterfix.split("C")[0])
                internal_C = int(afterfix.split("C")[1])
            except:
                print("auto extract N from conv_normN failed, set N to default 2")
                N = 0
                internal_C = 64
            decoder = conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                        input_dim=input_dim, output_dim=self.data_size,
                                                        activation=self.gan_AE_activation).cuda()
        elif "res_normN" in self.MIA_arch:
            try:
                afterfix = self.MIA_arch.split("res_normN")[1]
                N = int(afterfix.split("C")[0])
                internal_C = int(afterfix.split("C")[1])
            except:
                print("auto extract N from res_normN failed, set N to default 2")
                N = 0
                internal_C = 64
            decoder = res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                        input_dim=input_dim, output_dim=self.data_size,
                                                        activation=self.gan_AE_activation).cuda()
        
        else:
            raise ("No such GAN AE type.")

        '''Setting attacker's learning algorithm'''
        optimizer = torch.optim.Adam(decoder.parameters(), lr=MIA_lr)
        
        # Construct a dataset for training the decoder
        trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

        
        try:
            '''Use a fix set of testing image for each experiment'''
            images = torch.load(f"./saved_tensors/test_{self.dataset}_image.pt")
            labels = torch.zeros((images.size(0),)).long()
        except:
            '''Generate random images/activation pair:'''
            image_list = []
            for i, (images, _) in enumerate(self.train_loader):
                if i >= (128 // self.batch_size):
                    break
                image_list.append(images)
            images = torch.cat(image_list, dim = 0)
            torch.save(images, f"./saved_tensors/test_{self.dataset}_image.pt")
            labels = torch.zeros((images.size(0),)).long()

        self.save_image_act_pair(images, labels, 0)

        # Do real test on target's client activation (and test with target's client ground-truth.)
        sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}".format(
            target_client), self.save_dir + "/save_activation_client_{}".format(target_client))
        
        # Perform MIA Attack
        self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, path_dict, attack_batchsize)
        
        mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, sp_testloader, path_dict, attack_batchsize)

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)
        return mse_score, ssim_score, psnr_score
    

    def gen_ir(self, val_single_loader, local_model, img_folder="./tmp", intermed_reps_folder="./tmp"):
        """
        Generate (Raw Input - Intermediate Representation) Pair for Training of the AutoEncoder
        """
        # switch to evaluate mode
        local_model.eval()
        file_id = 0
        for i, (input, target) in enumerate(val_single_loader):
            # input = input.cuda(async=True)
            input = input.cuda()
            target = target.item()

            img_folder = os.path.abspath(img_folder)
            intermed_reps_folder = os.path.abspath(intermed_reps_folder)
            if not os.path.isdir(intermed_reps_folder):
                os.makedirs(intermed_reps_folder)
            if not os.path.isdir(img_folder):
                os.makedirs(img_folder)

            # compute output
            with torch.no_grad():
                ir = local_model(input)
            
            ir = ir.float()

            inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
            out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
            input = denormalize(input, self.dataset)
            save_image(input, inp_img_path)
            torch.save(ir.cpu(), out_tensor_path)
            file_id += 1
        print("Overall size of Training/Validation Datset for AE is {}: {}".format(int(file_id * 0.9),
                                                                                   int(file_id * 0.1)))


    def save_image_act_pair(self, input, target, client_id):
        """
            Run one train epoch
        """
        path_dir = os.path.join(self.save_dir, 'save_activation_client_{}'.format(client_id))
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
        else:
            rmtree(path_dir)
            os.makedirs(path_dir)
        input = input.cuda()

        for j in range(input.size(0)):
            img = input[None, j, :, :, :]
            label = target[None, j]
            with torch.no_grad():
                self.model.local_list[0].eval()
                save_activation = self.model.local_list[0](img)
            
            img = denormalize(img, self.dataset)
            
            save_activation = save_activation.float()
            
            save_image(img, os.path.join(path_dir, "{}.jpg".format(j)))
            torch.save(save_activation.cpu(), os.path.join(path_dir, "{}.pt".format(j)))
            torch.save(label.cpu(), os.path.join(path_dir, "{}.label".format(j)))

    def attack(self, num_epochs, decoder, optimizer, trainloader, testloader, path_dict, batch_size):
        round_ = 0
        min_val_loss = 999.
        train_output_freq = 10
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        # Optimize based on MSE distance
        criterion = nn.MSELoss()

        device = next(decoder.parameters()).device
        
        decoder.train()
        for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
            for num, data in enumerate(trainloader, 1):
                img, ir = data
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)
                
                output = decoder(ir)

                reconstruction_loss = criterion(output, img)
                train_loss = reconstruction_loss

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_losses.update(train_loss.item(), ir.size(0))

            if (epoch + 1) % train_output_freq == 0:
                save_images(img, output, epoch, path_dict["train_output_path"], offset=0, batch_size=batch_size)

            for num, data in enumerate(testloader, 1):
                img, ir = data

                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)

                output = decoder(ir)

                reconstruction_loss = criterion(output, img)
                val_loss = reconstruction_loss

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                
                val_losses.update(val_loss.item(), ir.size(0))

            # torch.save(decoder.state_dict(), path_dict["model_path"])
            self.logger.debug(
                "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f})".format(
                    epoch + 1,
                    num_epochs, train_losses=train_losses, val_losses=val_losses))
        self.logger.debug("Using MIA arch {} Best Validation Loss is {}".format(self.MIA_arch,min_val_loss))

    # This function means testing of the attacker's inversion model
    def test_attack(self, num_epochs, decoder, sp_testloader, path_dict, batch_size):
        device = next(decoder.parameters()).device
        new_state_dict = torch.load(path_dict["model_path"])
        decoder.load_state_dict(new_state_dict)
        decoder.eval()

        all_test_losses = AverageMeter()
        ssim_test_losses = AverageMeter()
        psnr_test_losses = AverageMeter()
        ssim_loss = pytorch_ssim.SSIM()

        criterion = nn.MSELoss()

        for num, data in enumerate(sp_testloader, 1):
            img, ir, label = data

            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)
            output_imgs = decoder(ir)
            reconstruction_loss = criterion(output_imgs, img)
            ssim_loss_val = ssim_loss(output_imgs, img)
            psnr_loss_val = get_PSNR(img, output_imgs)
            all_test_losses.update(reconstruction_loss.item(), ir.size(0))
            ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
            psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
            save_images(img, output_imgs, num_epochs, path_dict["test_output_path"], offset=num, batch_size=batch_size)

        self.logger.debug(
            "MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(all_test_losses.avg))
        self.logger.debug(
            "SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(ssim_test_losses.avg))
        self.logger.debug(
            "PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(psnr_test_losses.avg))
        return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg
    
def save_images(input_imgs, output_imgs, epoch, path, offset=0, batch_size=64): # saved image from tensor to jpg
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for img_idx in range(output_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, offset * batch_size + img_idx)

        if input_imgs is not None:
            save_image(input_imgs[img_idx], inp_img_path)
        if output_imgs is not None:
            save_image(output_imgs[img_idx], out_img_path)

def get_PSNR(refimg, invimg, peak = 1.0):
    psnr = 10*torch.log10(peak**2 / torch.mean((refimg - invimg)**2))
    return psnr

class ImageTensorFolder(torch.utils.data.Dataset):

    def __init__(self, img_path, tensor_path, label_path = "None", img_fmt="npy", tns_fmt="npy", lbl_fmt="npy", transform=None, limited_num = None):
        self.img_fmt = img_fmt
        self.tns_fmt = tns_fmt
        self.lbl_fmt = lbl_fmt
        select_idx = None
        if limited_num is not None:
            limited_num_10 = (limited_num// 10) * 10
            select_idx = []
            visited_label = {}
            filepaths = label_path + "/*.{}".format(lbl_fmt)
            files = sorted(glob(filepaths))
            count = 0
            index = 0
            # for index in range(limited_num_10):
            while count < limited_num_10:
                label = self.load_tensor(files[index], file_format=self.lbl_fmt)
                if label.item() not in visited_label:
                    visited_label[label.item()] = 1
                    select_idx.append(index)
                    # print(label.item())
                    count += 1
                elif visited_label[label.item()] < limited_num_10 // 10:
                    visited_label[label.item()] += 1
                    select_idx.append(index)
                    # print(label.item())
                    count += 1
                index += 1
                # print(label.item())
        self.img_paths = self.get_all_files(img_path, file_format=img_fmt, select_idx = select_idx)
        self.tensor_paths = self.get_all_files(tensor_path, file_format=tns_fmt, select_idx = select_idx)
        if label_path != "None":
            self.label_paths = self.get_all_files(label_path, file_format=lbl_fmt, select_idx = select_idx)
        else:
            self.label_paths = None
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def get_all_files(self, path, file_format="png", select_idx = None):
        filepaths = path + "/*.{}".format(file_format)
        files = sorted(glob(filepaths))
        # print(files[0:10])
        if select_idx is None:
            return files
        else:
            file_list = []
            for i in select_idx:
                file_list.append(files[i])
            return file_list

    def load_img(self, filepath, file_format="png"):
        if file_format in ["png", "jpg", "jpeg"]:
            img = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(img).shape[0] == 4:
                img = self.to_tensor(img)[:3, :, :]
                img = self.to_pil(img)
        elif file_format == "npy":
            img = np.load(filepath)
            #cifar10_mean = [0.4914, 0.4822, 0.4466]
            #cifar10_std = [0.247, 0.243, 0.261]
            img = np.uint8(255 * img)
            img = self.to_pil(img)
        elif file_format == "pt":
            img = torch.load(filepath)
        else:
            print("Unknown format")
            exit()
        return img

    def load_tensor(self, filepath, file_format="png"):
        if file_format == "png":
            tensor = Image.open(filepath)
            # Drop alpha channel
            if self.to_tensor(tensor).shape[0] == 4:
                tensor = self.to_tensor(tensor)[:3, :, :]
        elif file_format == "npy":
            tensor = np.load(filepath)
            tensor = self.to_tensor(tensor)
        elif file_format == "pt":
            tensor = torch.load(filepath)
            if len(tensor.size()) == 4:
                tensor = tensor.view(tensor.size()[1:])
            # print(tensor.size())
            tensor.requires_grad = False
        elif file_format == "label":
            tensor = torch.load(filepath)
            if len(tensor.size()) == 4:
                tensor = tensor.view(tensor.size()[1:])
            # print(tensor.size())
            tensor.requires_grad = False
        return tensor

    def __getitem__(self, index):
        img = self.load_img(self.img_paths[index], file_format=self.img_fmt)

        if self.transform is not None:
            img = self.transform(img)

        intermed_rep = self.load_tensor(self.tensor_paths[index], file_format=self.tns_fmt)

        if self.label_paths is not None:
            label = self.load_tensor(self.label_paths[index], file_format=self.lbl_fmt)
            return img, intermed_rep, label
        else:
            return img, intermed_rep

    def __len__(self):
        return len(self.img_paths)

from torch.utils.data import SubsetRandomSampler

def apply_transform_test(batch_size, image_data_dir, tensor_data_dir, limited_num = None, shuffle_seed = 123, dataset = None):
    """
    """
    std = [1.0, 1.0, 1.0]
    mean = [0.0, 0.0, 0.0]

    trainTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])
    dataset = ImageTensorFolder(img_path=image_data_dir, tensor_path=tensor_data_dir, label_path=tensor_data_dir,
                                 img_fmt="jpg", tns_fmt="pt", lbl_fmt="label", transform=trainTransform, limited_num = limited_num)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return testloader

def apply_transform(batch_size, image_data_dir, tensor_data_dir, shuffle_seed = 123, dataset = None):
    """
    """
    std = [1.0, 1.0, 1.0]
    mean = [0.0, 0.0, 0.0]
    
    train_split = 0.9
    trainTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)
                                         ])
    dataset = ImageTensorFolder(img_path=image_data_dir, tensor_path=tensor_data_dir,
                                 img_fmt="jpg", tns_fmt="pt", transform=trainTransform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    np.random.seed(shuffle_seed)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=4,
                                              sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             sampler=test_sampler)
    return trainloader, testloader

class WSConv2d(nn.Conv2d): # This module is taken from https://github.com/joe-siyuan-qiao/WeightStandardization

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class WSTransposeConv2d(nn.ConvTranspose2d): # This module is taken from https://github.com/joe-siyuan-qiao/WeightStandardization

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=1, groups=1, bias=True, dilation=1):
        super(WSTransposeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias, dilation)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv_transpose2d(x, weight, self.bias, self.stride,
                        self.padding, self.output_padding, self.groups, self.dilation)


class custom_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class conv_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid", GN_option = False, WS = True):
        super(conv_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
        if GN_option and WS:
            model += [WSConv2d(input_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
        else:
            model += [nn.Conv2d(input_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #first
        if not GN_option:
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        model += [nn.ReLU()]

        for _ in range(N):
            if GN_option and WS:
                model += [WSConv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
            else:
                model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #Middle-N
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            if GN_option and WS:
                model += [WSTransposeConv2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            else:
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        else:
            if GN_option and WS:
                model += [WSConv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
            else:
                model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #two required
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            if GN_option and WS:
                model += [WSTransposeConv2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            else:
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        else:
            if GN_option and WS:
                model += [WSConv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)]
            else:
                model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #two required
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                if GN_option and WS:
                    model += [WSTransposeConv2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                else:
                    model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                if not GN_option:
                    model += [nn.BatchNorm2d(internal_nc)]
                else:
                    model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
                model += [nn.ReLU()]
        if GN_option and WS:
            model += [WSConv2d(internal_nc, output_nc, kernel_size=3, stride=1, padding=1)] #last
        else:
            model += [nn.Conv2d(internal_nc, output_nc, kernel_size=3, stride=1, padding=1)] #last
        if not GN_option:
            model += [nn.BatchNorm2d(output_nc)]
        else:
            model += [nn.GroupNorm(max(output_nc//NUM_CHANNEL_GROUP, 1), output_nc)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1, GN_option = False, WS = True):
        super(ResBlock, self).__init__()
        self.bn = bn
        self.GN_option = GN_option
        if GN_option:
            self.bn0 = nn.GroupNorm(max(in_planes//NUM_CHANNEL_GROUP, 1), in_planes)
        elif bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        if GN_option and WS:
            self.conv1 = WSConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if GN_option:
            self.bn1 = nn.GroupNorm(max(planes//NUM_CHANNEL_GROUP, 1), planes)
        elif bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn or self.GN_option:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn or self.GN_option:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


class res_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid", GN_option = False, WS = True):
        super(res_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
            
        model += [ResBlock(input_nc, internal_nc, bn = True, stride=1, GN_option = GN_option, WS = WS)] #first
        model += [nn.ReLU()]

        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1, GN_option = GN_option, WS = WS)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            if GN_option and WS:
                model += [WSTransposeConv2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            else:
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1, GN_option = GN_option, WS = WS)] #second
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            if GN_option and WS:
                model += [WSTransposeConv2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            else:
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if not GN_option:
                model += [nn.BatchNorm2d(internal_nc)]
            else:
                model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1, GN_option = GN_option, WS = WS)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                if GN_option and WS:
                    model += [WSTransposeConv2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                else:
                    model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                if not GN_option:
                    model += [nn.BatchNorm2d(internal_nc)]
                else:
                    model += [nn.GroupNorm(max(internal_nc//NUM_CHANNEL_GROUP, 1), internal_nc)]
                model += [nn.ReLU()]

        model += [ResBlock(internal_nc, output_nc, bn = True, stride=1, GN_option = GN_option, WS = WS)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output


