# external.py ---
#
# Filename: external.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:53:54 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary: Codes from other places. See LICENSES for more details
#
#
#
#

# Change Log:
#
#
#

# Code:

import pickle


def unpickle(file_name):
    """unpickle function from CIFAR10 webpage"""
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


#
# external.py ends here
