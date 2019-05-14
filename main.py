# main.py ---
#
# Filename: main.py
# Maintainer: Anmol Mann

# Description: for training and testing the model
# Course Instructor: Kwang Moo Yi
# Version: v.1.0
# Package-Requires: See requirements.txt
# URL: 
# Doc URL:
# Keywords: GANs, PatchGAN, WGAN, AUs, etc.
# Compatibility: Python3.7.0
#
#

# Commentary:
# A GAN conditioning scheme based on Action Units (AU) annotations, 
# which describe in a continuous manifold the anatomical facial movements 
# defining a human expression. 
# Reference: https://arxiv.org/abs/1807.09251

import os, random, imageio

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import torchvision.transforms as transforms
from torchvision.utils import save_image as torch_save

from config import get_config, print_usage
from model import Generate_GAN, Discriminate_GAN
from tensorboardX import SummaryWriter
from utils.faces import load_data
from utils.datawrapper import FacesDataset

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cl_loss():
    """
    Returns the classification loss object
    """

    # Changed it to BCE but no effect on the results
    if torch.cuda.is_available():
        return torch.nn.MSELoss().cuda()
    else:
        return torch.nn.MSELoss()


def identity_loss():
    """
    return identity loss function
    """
    if torch.cuda.is_available():
        return torch.nn.L1Loss().cuda()
    else:
        return torch.nn.L1Loss()

def sat_att_loss():
    """
    Returns the attention saturation loss
    :return:
    """
    if torch.cuda.is_available():
        return torch.nn.MSELoss().cuda()
    else:
        return torch.nn.MSELoss()

def train(config):
    """Training process.

    """

    # Initialize datasets for both training and validation
    train_data = FacesDataset(
        config, mode="train",
    )

    # Create data loader for training and validation.
    tr_data_loader = DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=True
    )

    # Create model instance.
    G = Generate_GAN(
        config.nchannel_base_g, config.dim_labels, config.num_resnet_g
    )
    D = Discriminate_GAN(
        config.trimg_size, config.nchannel_base_d, config.dim_labels, config.num_resnet_d
    )

    if torch.cuda.is_available():
        # distribute training
        torch.nn.DataParallel(
        	D, 
        	device_ids=list(range(torch.cuda.device_count()))
        )
        torch.nn.DataParallel(
        	G, 
        	device_ids=list(range(torch.cuda.device_count()))
        )
        G = G.cuda()
        D = D.cuda()

    # Make sure that the model is set for training
    G.train()
    D.train()

    # Create loss objects
    cl_loss_ = cl_loss()
    identity_loss_ = identity_loss()
    sat_att_loss_ = sat_att_loss()

    if torch.cuda.is_available():
        torch.nn.DataParallel(cl_loss_, device_ids=list(range(torch.cuda.device_count())))
        torch.nn.DataParallel(identity_loss_, device_ids=list(range(torch.cuda.device_count())))
        torch.nn.DataParallel(sat_att_loss_, device_ids=list(range(torch.cuda.device_count())))

    # Initialize lr for decaying.
    lr_generator = config.lr_generator
    lr_discriminator = config.lr_discriminator

    # Create optimizier
    # we set up two separate optimizers, one for D and one for G
    gen_optimizer = optim.Adam(G.parameters(), lr = config.lr_generator, betas = (config.beta1, config.beta2))
    dis_optimizer = optim.Adam(D.parameters(), lr = config.lr_discriminator, betas = (config.beta1, config.beta2))
    # No need to move the optimizer (as of PyTorch 1.0), it lies in the same
    # space as the model

    # Create summary writer
    tr_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train")
    )

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize training
    iter_idx = -1  # make counter start at zero

    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")

    # Check for existing training results. 
    if os.path.exists(checkpoint_file):
        if config.resume:
            
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            # reference: https://pytorch.org/docs/stable/torch.html?highlight=load#torch.load
            load_res = torch.load(checkpoint_file, map_location = lambda storage, loc: storage)
            # Resume iterations
            iter_idx = load_res["iter_idx"]
            # Resume generator and discriminator
            G.load_state_dict(load_res["generator"])
            D.load_state_dict(load_res["discriminator"])
            # Resume optimizer
            gen_optimizer.load_state_dict(load_res["gen_optimizer"])
            dis_optimizer.load_state_dict(load_res["dis_optimizer"])
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in range(config.num_epoch):
        # For each iteration
        prefix = "Training Epoch {:3d}: ".format(epoch)

        for data in tqdm(tr_data_loader, desc=prefix):
        # for batch_idx, batch in enumerate(tr_data_loader):
            # Counter
            iter_idx += 1

            # Split the data
            # x = input images
            # y = Labels for computing classification loss.
            x, y = data   

            # Pre-process input data
            desired_exp = []
            for index_ in range(config.batch_size):
                random_num = random.randint(0, len(tr_data_loader) - 1)
                # Select a random AU vector from the set of image ids (label_curr)
                desired_exp_aux = tr_data_loader.dataset[random_num][1]
                # Apply a variance of 0.1 to the vector (Adding noise as taught in the lectures)
                desired_exp.append(
                    desired_exp_aux.numpy() + np.random.uniform(-0.1, 0.1, desired_exp_aux.shape)
                )

            desired_exp = torch.FloatTensor(desired_exp).clamp(0, 1)

            """  
            from PIL import Image
            img_name = random.choice(train_data.data)
            img_id = str(os.path.splitext(os.path.basename(img_name))[0])

            test_img = Image.open(img_name).convert("RGB")
            # make pytorch object and normalize it
            test_img_tensor = train_data.list_trans(test_img)
            # print(test_img_tensor)
            desired_exp = train_data.label[img_id]/5.0
            # Add noise
            desired_exp = torch.FloatTensor(desired_exp + np.random.uniform(-0.1, 0.1, desired_exp.shape))
            """

            if torch.cuda.is_available():
                # Original image labels (AU vectors)
                orig_exp = (y.clone()).cuda()
                # real input images
                x = x.cuda()
                # real AU vectors
                y = y.cuda()
                desired_exp = desired_exp.cuda()
                
                """
                # Target domain labels
                des_exp = (desired_exp.clone()).cuda()
                # Labels for computing classification loss.
                desired_exp = desired_exp.cuda()
            else:
                orig_exp = y.clone()
                des_exp = desired_exp.clone()
                """

            # Train the DISCRIMINATOR

            # COMPUTE VARIOUS MODEL LOSSES

            #### Had to implement the whole thing twice.
            # the whole first implmentation is commented after this piece of code
            # as it did not generate good results, so implemented the project again
            # as shown in PART 1, 2, and 3.
            # This new implementaiton, i must say, is bit cleaner and easy to relate than the previous one.

            #### PART 1, generate the fake image and then render back the original img
            # Bidirectional generator

            A_mask, C_mask = G.forward(x, desired_exp)
            fake_img_ = (A_mask * x) + ((1 - A_mask) * C_mask)
            A_render_back, C_render_back = G.forward(fake_img_, y)
            real_img_render = (A_render_back * fake_img_) + ((1 - A_render_back) * C_render_back)

            #### PART 2, train discriminator first

            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            eval_x, eval_AU = D.forward(x)
            wgan_x_D_loss = -torch.mean(eval_x)
            AU_x_D_loss = cl_loss_(eval_AU, y)

            eval_fake, _ = D.forward(fake_img_.detach())
            wgan_fake_D_loss = torch.mean(eval_fake)

            # hyper-parameter Alpha for interpolation
            if torch.cuda.is_available():
                alpha = torch.rand(x.size(0), 1, 1, 1).cuda()
            else:
                alpha = torch.rand(x.size(0), 1, 1, 1)
            cur_aus_img = (x.data * alpha) + (fake_img_.data * (1 - alpha))
            cur_aus_img = cur_aus_img.requires_grad_(True)
            eval_IP_imgs, _ = D.forward(cur_aus_img)

            if torch.cuda.is_available():
                init_weight_ = torch.ones(eval_IP_imgs.size()).cuda()
            else:
                init_weight_ = torch.ones(eval_IP_imgs.size())

            grad_penalty_ = torch.autograd.grad(
                outputs = eval_IP_imgs,
                inputs = cur_aus_img,
                grad_outputs = init_weight_,
                retain_graph = True,
                create_graph = True,
                only_inputs = True
                )[0]

            grad_penalty_ = grad_penalty_.view(grad_penalty_.size(0), -1)
            # print("derivative:\t", grad_penalty_)
            grad_penalty_l2norm = torch.sqrt(torch.sum(grad_penalty_ ** 2, dim=1))
            # print("l2norm:\t", grad_penalty_)
            grad_pen_D = torch.mean((grad_penalty_l2norm - 1) ** 2)

            total_loss_D = (wgan_x_D_loss + wgan_fake_D_loss) * config.lambda_D
            total_loss_D = total_loss_D + (config.lambda_cls * AU_x_D_loss)
            total_loss_D = total_loss_D + (config.lambda_wgangp * grad_pen_D)

            # Creat loss dicts for logs folder.
            loss = {}
            loss['D/total_loss_D'], loss['D/AU_x_D_loss'] = total_loss_D.item(), config.lambda_cls * AU_x_D_loss.item()
            loss['D/wgan_fake_D_loss'], loss['D/wgan_x_D_loss'] = config.lambda_D * wgan_fake_D_loss.item(), config.lambda_D * wgan_x_D_loss.item()
            loss['D/grad_pen_D'] = config.lambda_wgangp * grad_pen_D

            # compute gradients
            total_loss_D.backward()
            # update parameters
            dis_optimizer.step()

            #### PART 3, train the generator

            if (iter_idx + 1) % config.critic_balance == 0:

                # Zero the parameter gradients in the optimizer (reset gradients)
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()

                eval_fake_G, eval_fake_AU_G = D.forward(fake_img_)
                wgan_fake_G_loss = -torch.mean(eval_fake_G)
                AU_fake_G_loss = cl_loss_(eval_fake_AU_G, desired_exp)
                iden_G_loss = identity_loss_(real_img_render, x)

                # Attention Masks, lambda_masks = 0 (default)
                AU_mask_G_loss = config.lambda_masks * (torch.mean(A_mask) + torch.mean(A_render_back))

                # Smoothness Loss, lambda_sm = 0 (default)
                # Total Variation Loss
                sm_loss_1_1 = torch.mean(torch.abs(A_mask[:, :, :, :-1] - A_mask[:, :, :, 1:]) ** 2)
                sm_loss_1_2 = torch.mean(torch.abs(A_mask[:, :, :-1, :] - A_mask[:, :, 1:, :]) ** 2)
                sm_loss_1 = torch.mean(sm_loss_1_1 + sm_loss_1_2)

                sm_loss_2_1 = torch.mean(torch.abs(A_render_back[:, :, :, :-1] - A_render_back[:, :, :, 1:]) ** 2)
                sm_loss_2_2 = torch.mean(torch.abs(A_render_back[:, :, :-1, :] - A_render_back[:, :, 1:, :]) ** 2)
                sm_loss_2 = torch.mean(sm_loss_2_1 + sm_loss_2_2)
                
                AU_smooth_G_loss = config.lambda_sm * (sm_loss_1 + sm_loss_2)

                total_loss_G = wgan_fake_G_loss * config.lambda_D
                total_loss_G = total_loss_G + (config.lambda_cls * AU_fake_G_loss)
                total_loss_G = total_loss_G + (config.lambda_rec * iden_G_loss) + AU_mask_G_loss 
                total_loss_G = total_loss_G + AU_smooth_G_loss

                loss['G/total_loss_G'], loss['G/wgan_fake_G_loss'] = total_loss_G.item(), wgan_fake_G_loss.item() * config.lambda_D
                loss['G/AU_fake_G_loss'] = AU_fake_G_loss.item() * config.lambda_cls
                loss['G/iden_G_loss'], loss['G/AU_mask_G_loss'] = iden_G_loss.item() * config.lambda_rec, AU_mask_G_loss
                loss['G/AU_smooth_G_loss'] = AU_smooth_G_loss.item()

                # compute gradients
                total_loss_G.backward()
                # update parameters
                gen_optimizer.step()

            ####

            # This implementation below is the one I implemented t first and was talking 
            # about it above as well. This did not give me good results.
            # So, had to scratch it off.

            """

            # loss calculation for real images
            Img_eval_real, AU_eval_real = D.forward(x)
            # print(y.shape, Img_eval_real.shape, AU_eval_real.shape)
            # wgan-gp
            loss_dis_real = -torch.mean(Img_eval_real)
            loss_dis_cls = cl_loss_(AU_eval_real, y) / config.batch_size

            # loss calculation for fake images

            # compute fake image from G first
            attmask_fake, colormask_fake = G.forward(x, des_exp)

            # from Reference paper and code: Pumarola [GANimation]
            
            merge_masks_G = (1 - attmask_fake) * colormask_fake + attmask_fake * x
            Img_eval_real, _ = D(merge_masks_G.detach())
            dfake_imloss_gen = torch.mean(Img_eval_real)

            if torch.cuda.is_available():
                # Loss calculation (Penalizing gradient: (L2_norm(dy/dx) - 1)**2)
                alpha = torch.rand(x.size(0), 1, 1, 1).cuda()
            else:
                alpha = torch.rand(x.size(0), 1, 1, 1)
            # control alpha parameter for generating ouput expression
            merge_masks = (alpha * x.data + (1 - alpha) * merge_masks_G.data).requires_grad_(True)
            Img_eval_real, _ = D(merge_masks)

            if torch.cuda.is_available():
                init_weight_ = torch.ones(Img_eval_real.size()).cuda()
            else:
                init_weight_ = torch.ones(Img_eval_real.size())

            grad_penalty_ = torch.autograd.grad(
                outputs=Img_eval_real,
                inputs=merge_masks,
                grad_outputs=init_weight_,
                retain_graph=True,
                create_graph=True,
                only_inputs=True
                )[0]

            grad_penalty_ = grad_penalty_.view(grad_penalty_.size(0), -1)
            # print("derivative:\t", grad_penalty_)
            grad_penalty_l2norm = torch.sqrt(torch.sum(grad_penalty_ ** 2, dim=1))
            # print("l2norm:\t", grad_penalty_)
            wgangp_dis_loss = torch.mean((grad_penalty_l2norm - 1) ** 2)

            # Backward and optimize.
            # Compute Total Loss
            loss_dis = loss_dis_real + dfake_imloss_gen + (config.lambda_cls * loss_dis_cls) + (config.lambda_wgangp * wgangp_dis_loss)

            # Zero the parameter gradients in the optimizer (reset gradients)
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()
            # compute gradients
            loss_dis.backward()
            # update parameters
            dis_optimizer.step()

            # Creat loss dicts for logs folder.
            loss = {}
            loss['D/loss'], loss['D/real_imloss_gen'] = loss_dis.item(), loss_dis_real.item()
            loss['D/fake_imloss_gen'], loss['D/loss_cls'] = dfake_imloss_gen.item(), config.lambda_cls * loss_dis_cls.item()
            loss['D/loss_wgangp'] = config.lambda_wgangp * wgangp_dis_loss.item()

            # Train the GENERATOR
            if (iter_idx + 1) % config.critic_balance == 0:
                # generate estimated img from input image (Mapping)
                attmask_fake, colormask_fake = G(x, des_exp)

                merge_masks_G = (1 - attmask_fake) * colormask_fake + attmask_fake * x
                Img_eval_real, AU_eval_real = D(merge_masks_G)
                wganGAN_fake_loss = -torch.mean(Img_eval_real)
                loss_gen_cls = cl_loss_(AU_eval_real, desired_exp) / config.batch_size

                # render back original input from estimated image
                real_Att_second, real_C_second = G(merge_masks_G, orig_exp)

                if config.mask_for_sat:
                    real_Att_second = torch.clamp(3 * torch.tanh(real_Att_second - 0.5) + 0.5, 0, 1)
                real_second_time = (1 - real_Att_second) * real_C_second + real_Att_second * x
                G_second_loss = identity_loss_(x, real_second_time)

                if torch.cuda.is_available():
                    G_fake_AUmask_loss_ = config.lambda_sat * sat_att_loss_(
                        attmask_fake,
                        torch.zeros(attmask_fake.size()).cuda()
                    )

                    G_real_AUmask_loss_ = config.lambda_sat * sat_att_loss_(
                        real_Att_second,
                        torch.zeros(real_Att_second.size()).cuda()
                    )
                else:
                    G_fake_AUmask_loss_ = config.lambda_sat * sat_att_loss_(
                        attmask_fake,
                        torch.zeros(attmask_fake.size())
                    )

                    G_real_AUmask_loss_ = config.lambda_sat * sat_att_loss_(
                        real_Att_second,
                        torch.zeros(real_Att_second.size())
                    )

                sm_loss_1_1 = torch.mean(torch.abs(attmask_fake[:, :, :, :-1] - attmask_fake[:, :, :, 1:]) ** 2)
                sm_loss_1_2 = torch.mean(torch.abs(attmask_fake[:, :, :-1, :] - attmask_fake[:, :, 1:, :]) ** 2)
                sm_loss_1 = torch.mean(sm_loss_1_1 + sm_loss_1_2)
                smooth_loss_G_fake = config.lambda_sm * sm_loss_1

                sm_loss_2_1 = torch.mean(torch.abs(real_Att_second[:, :, :, :-1] - real_Att_second[:, :, :, 1:]) ** 2)
                sm_loss_2_2 = torch.mean(torch.abs(real_Att_second[:, :, :-1, :] - real_Att_second[:, :, 1:, :]) ** 2)
                sm_loss_2 = torch.mean(sm_loss_2_1 + sm_loss_2_2)
                smooth_loss_G_real = config.lambda_sm * sm_loss_2

                # Total attention (smoothness) Loss
                loss_Att_G = smooth_loss_G_fake + smooth_loss_G_real + G_fake_AUmask_loss_ + G_real_AUmask_loss_

                # Backward and optimize; Compute Total Loss
                loss_gen = (config.lambda_rec * G_second_loss) + (config.lambda_cls * loss_gen_cls) + loss_Att_G + wganGAN_fake_loss
                
                # Zero the parameter gradients in the optimizer (reset gradients)
                gen_optimizer.zero_grad()
                dis_optimizer.zero_grad()
                # compute gradients
                loss_gen.backward()
                # update parameters
                gen_optimizer.step()

                # update loss dict for logging.
                loss['G/loss_wganGAN_fake'] = wganGAN_fake_loss.item()
                loss['G/loss_second'] = config.lambda_rec * G_second_loss.item()
                loss['G/loss_cls'] = config.lambda_cls * loss_gen_cls.item()
                loss['G/loss_attention'] = loss_Att_G.item()
                loss['G/loss_smooth_fake'], loss['G/loss_smooth_real'] = smooth_loss_G_fake.item(), smooth_loss_G_real.item()
                loss['G/loss_sat_fake'], loss['G/loss_sat_real'] = G_fake_AUmask_loss_.item(), G_real_AUmask_loss_.item()
                loss['G/loss'] = loss_gen.item()

                # print(loss)
                """

            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:

                # Write loss to tensorboard, using keywords `loss`
                for key, loss_value in loss.items():
                    tr_writer.add_scalar(key, loss_value, global_step = iter_idx)
                    print(key, "\t", loss_value)
                # Save
                torch.save({
                    "iter_idx": iter_idx,
                    "generator": G.state_dict(),
                    "discriminator": D.state_dict(),
                    "gen_optimizer": gen_optimizer.state_dict(),
                    "dis_optimizer": dis_optimizer.state_dict()
                }, checkpoint_file)

        # Decay learning rates.
        if (epoch + 1) > config.num_epochs_decay:
            lr_generator -= (config.lr_generator / 10.0)  # float(self.num_epochs_decay))
            lr_discriminator -= (config.lr_discriminator / 10.0)  # float(self.num_epochs_decay))
            # https://stackoverflow.com/questions/48324152/pytorch-how-to-change-the-learning-rate-of-an-optimizer-at-any-given-moment-no
            # change learning rates of the generator and discriminator after a point
            for g_par in gen_optimizer.param_groups:
                g_par['lr'] = lr_generator
            for d_par in dis_optimizer.param_groups:
                d_par['lr'] = lr_discriminator
            print ('Decayed learning rates, lr_generator: {}, lr_discriminator: {}.'.format(lr_generator, lr_discriminator))

def save_numpy_img(np_img):
    from PIL import Image
    # the results are saved in the results dir
    # As I wanted to tile the results together as done in the
    # orignal paper by the author, so converted the imgs to that size
    # first and then glued/concatenated them together
    if np_img.shape[0] == 1:
        np_img = np.tile(np_img, (3, 1, 1))  
    np_img = (np.transpose(np_img, (1, 2, 0)) / 2. + 0.5) * 255.0
    np_img = Image.fromarray(np_img.astype(np.uint8))
    return np_img

def test(config):
    """Test routine"""

    # Initialize Dataset for testing.
    test_data = FacesDataset(
        config, mode="test",
    )

    # Create data loader for the test dataset with 4 number of workers and no
    # shuffling.
    te_data_loader = DataLoader(
        dataset=test_data,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=False
    )

    # Create model
    G = Generate_GAN(
        config.nchannel_base_g, config.dim_labels, config.num_resnet_g
    )
    D = Discriminate_GAN(
        config.trimg_size, config.nchannel_base_d, config.dim_labels, config.num_resnet_d
    )
    # move to GPU
    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()

    # Load our best model and set model for testing
    # reference: https://pytorch.org/docs/stable/torch.html?highlight=load#torch.load
    load_res = torch.load(
        os.path.join(config.save_dir, "checkpoint.pth"), map_location = lambda storage, loc: storage
    )
    G.load_state_dict(load_res["generator"])
    D.load_state_dict(load_res["discriminator"])

    """
    pretrained_dict = {k: v for k, v in load_gen.items() if k in G.state_dict()}
    for k, v in load_gen.items():
        print(k)
    G.load_state_dict(pretrained_dict)
    """

    G.eval()
    D.eval()

    # Implement The Test loop
    prefix = "Testing: "
    te_loss = []
    index_result = 0
    for batch_idx, batch in enumerate(te_data_loader):

        # Split the data
        x, y = batch

        # Send data to GPU if we have one
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

	    # Don't invoke gradient computation
        with torch.no_grad():
            out_test_frame = [x.float().numpy()]

            from PIL import Image

            img_name = random.choice(test_data.data)
            img_id = str(os.path.splitext(os.path.basename(img_name))[0])

            test_img = Image.open(img_name).convert("RGB")
            # make pytorch object and normalize it
            test_img_tensor = test_data.list_trans(test_img)
            # print(test_img_tensor)
            desired_AU = torch.FloatTensor(test_data.label[img_id]/5.0)

            # No. of fake imgs to produce
            # Change this '5' here in range() to generate different 
            # number of interpolation imgs.
            for idx in range(5):
            	# control hyper-parameter alpha
                hyper_alpha = (idx + 1.) / float(5)
                cur_tar_aus = hyper_alpha * desired_AU + (1 - hyper_alpha) * y
                # cur_tar_aus = hyper_alpha * np.random.uniform(0, 1, size=(config.batch_size, 17))
                # cur_tar_aus = torch.from_numpy(cur_tar_aus/5.0).float()
                # print(y.shape, "--", cur_tar_aus.shape)
                if torch.cuda.is_available():
                	cur_tar_aus.cuda()
                	test_img_tensor.cuda()
                # print(cur_tar_aus)
                # test_batch = {'src_img': x, 'tar_aus': cur_tar_aus, 'src_aus':y, 'tar_img':test_img_tensor}

                # Fake image Generation
                attmask_fake, colormask_fake = G(x, cur_tar_aus)
                fake_img = attmask_fake * x + (1 - attmask_fake) * colormask_fake

                # print(fake_img)                
                out_test_frame.append(fake_img.cpu().float().numpy())

            out_test_frame.append(test_img_tensor.float().numpy())
            # print(test_img_tensor.shape, " ---- ", len(out_test_frame))
            # print(len(batch[0]))

        length_inter = len(out_test_frame) - 1

        for frame in range(len(batch[0])):
            if not config.as_gif:
                animation_frames = np.array(save_numpy_img(out_test_frame[0][frame]))
                for animation_num in range(1, length_inter):
                    print(animation_frames.shape, out_test_frame[animation_num][frame].shape)
                    temp_out_test_frame = np.array(save_numpy_img(out_test_frame[animation_num][frame]))
                    # ValueError: axes don't match array
                    animation_frames = np.concatenate((animation_frames, temp_out_test_frame), axis=1)
                animation_frames = Image.fromarray(animation_frames)
                # save static image
                output_path = os.path.join(config.result_dir, "{}_.jpg".format(index_result))
                animation_frames.save(output_path)
            else:
                imgs_frames = []
                for animation_num in range(length_inter):
                    temp_out_test_frame = np.array(save_numpy_img(out_test_frame[animation_num][frame]))
                    imgs_frames.extend([temp_out_test_frame for _ in range(3)])
                # save gif image
                output_path = os.path.join(config.result_dir, "{}_.gif".format(index_result))
                imageio.mimsave(output_path, imgs_frames)
                
            index_result += 1

            print("Testing _{}_ Done!".format(idx))

def main(config):
    """The main function."""

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


#
# main.py ends here
