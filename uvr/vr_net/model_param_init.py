import json

default_param = {}
default_param["bins"] = -1
default_param["unstable_bins"] = -1  # training only
default_param["stable_bins"] = -1  # training only
default_param["sr"] = 44100
default_param["pre_filter_start"] = -1
default_param["pre_filter_stop"] = -1
default_param["band"] = {}

N_BINS = "n_bins"


def int_keys(d):
    """
    Converts string keys that represent integers into actual integer keys in a list.

    This function is particularly useful when dealing with JSON data that may represent
    integer keys as strings due to the nature of JSON encoding. By converting these keys
    back to integers, it ensures that the data can be used in a manner consistent with
    its original representation, especially in contexts where the distinction between
    string and integer keys is important.

    Args:
        input_list (list of tuples): A list of (key, value) pairs where keys are strings
                                     that may represent integers.

    Returns:
        dict: A dictionary with keys converted to integers where applicable.
    """
    # Initialize an empty dictionary to hold the converted key-value pairs.
    result_dict = {}
    # Iterate through each key-value pair in the input list.
    for key, value in d:
        # Check if the key is a digit (i.e., represents an integer).
        if key.isdigit():
            # Convert the key from a string to an integer.
            key = int(key)
        result_dict[key] = value
    return result_dict


class ModelParameters(object):
    """
    A class to manage model parameters, including loading from a configuration file.

    Attributes:
        param (dict): Dictionary holding all parameters for the model.
    """

    def __init__(self, config_path=""):
        """
        Initializes the ModelParameters object by loading parameters from a JSON configuration file.

        Args:
            config_path (str): Path to the JSON configuration file.
        """

        # Load parameters from the given configuration file path.
        with open(config_path, "r") as f:
            self.param = json.loads(f.read(), object_pairs_hook=int_keys)

        # Ensure certain parameters are set to False if not specified in the configuration.
        for k in [
            "mid_side",
            "mid_side_b",
            "mid_side_b2",
            "stereo_w",
            "stereo_n",
            "reverse",
        ]:
            if not k in self.param:
                self.param[k] = False

        # If 'n_bins' is specified in the parameters, it's used as the value for 'bins'.
        if N_BINS in self.param:
            self.param["bins"] = self.param[N_BINS]
