# config.py ---
#
# Filename: config.py
# Maintainer: Anmol Mann
# Description:
# Course Instructor: Kwang Moo Yi

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

main_arg.add_argument("--resize", type=bool,
        default=False,
        help="Crop images or not")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="celebA",
                       help="Directory with celeb data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=25,
                       help="Size of each training batch")

#edit
train_arg.add_argument("--lr_generator", type=float, default=0.0001,
                       help="Generator's Learning rate")
train_arg.add_argument("--lr_discriminator", type=int,
                       default=0.0001,
                       help="Discriminator's Learning rate")
train_arg.add_argument("--beta1", type=int,
                       default=0.5,
                       help="Size of each training batch")
train_arg.add_argument("--beta2", type=int,
                       default=0.999,
                       help="Size of each training batch")
train_arg.add_argument("--d_repeat_num", type=int,
                       default=6,
                       help="# of (strided) convolution layers in D")
train_arg.add_argument("--image_size", type=int,
                       default=128,
                       help="scale img to this size")
train_arg.add_argument("--g_conv_dim", type=int,
                       default=64,
                       help="conv filters in G")
train_arg.add_argument("--d_conv_dim", type=int,
                       default=64,
                       help="conv filters in D")
train_arg.add_argument("--c_dim", type=int,
                       default=4,
                       help="Size of AUs")
train_arg.add_argument('--lambda_wgangp',
                       type=float,
                       default=10.0,
                       help='weight for gradient penalty')
train_arg.add_argument('--lambda_cls', type=float,
                       default=160.0,
                       help='weight for domain classification loss')
train_arg.add_argument('--lambda_D', type=float,
                       default=1.0,
                       help='D weight')
train_arg.add_argument('--mask_for_sat',
                       action="store_true",
                       default=False,
                       help='Saturate the attention mask')
train_arg.add_argument('--lambda_sm',
                       type=float,
                       default=0,
                       help='weight for the attention smoothing loss')
train_arg.add_argument('--critic_balance',
                       type=int,
                       default=1, # change it to 5
                       help='number of D updates per each G update')
train_arg.add_argument('--lambda_sat',
                       type=float,
                       default=0.1,
                       help='weight for attention saturation loss')
train_arg.add_argument('--lambda_rec',
                       type=float,
                       default=10.0,
                       help='weight for reconstruction loss')
train_arg.add_argument('--lambda_masks',
                       type=float,
                       default=0.0,
                       help='weight for attention masks')
train_arg.add_argument('--num_epochs_decay',
                       type=int,
                       default=20,
                       help='number of epochs for start decaying lr')

# Model Arguments
model_arg = add_argument_group("Model Configuration")

model_arg.add_argument("--nchannel_base_g", type=int,
                       default=64,
                       help="Base number of channels of G")

model_arg.add_argument("--num_resnet_g", type=int,
                       default=6,
                       help="number of ResNet Blocks in G")

model_arg.add_argument("--num_resnet_d", type=int,
                       default=6,
                       help="number of ResNet Blocks in D")

model_arg.add_argument("--nchannel_base_d", type=int,
                       default=64,
                       help="Base number of channels of D")

model_arg.add_argument("--dim_labels", type=int,
                       default=17,
                       help="AU dimensions")

model_arg.add_argument("--trimg_size", type=int,
                       default=128,
                       help="Size of the input image for training")
# stop edit

train_arg.add_argument("--num_epoch", type=int,
                       default=25,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--result_dir", type=str,
                       default="./results",
                       help="Directory to save the test images")

train_arg.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")

train_arg.add_argument('--as_gif', type=str2bool,
  default=False,
  help='save gif images'
  )

def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
