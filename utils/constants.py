"""
Module to handle universal/general constants used across files.
"""

################################################################################
# Constants #
################################################################################

# GENERAL CONSTANTS:
SMALL_NUMBER = 1e-8
VERY_SMALL_NUMBER = 1e-12
SUPER_SMALL_NUMBER = 1e-24
INF = 1e20


_PAD_TOKEN = '#pad#'
_UNK_TOKEN = '<unk>'
_SOS_TOKEN = '<s>'
_EOS_TOKEN = '</s>'


# LOG FILES ##
_CONFIG_FILE = "config.json"
_SAVED_WEIGHTS_FILE = "params.saved"
_PREDICTION_FILE = "test_pred.txt"
_REFERENCE_FILE = "test_ref.txt"