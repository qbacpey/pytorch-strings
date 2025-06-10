from numpy.dtypes import StringDType
import torch

string_column = ["this is a longer string", "short string"]
np_arr = np.array(string_column, dtype=StringDType())
np_arr
#--> array(['this is a longer string', 'short string'], dtype=StringDType())

# !No String data-type in PyTorch!

def do_plain_encoding(np_arr):
    # padding each str to the same max_length
    padded_arr = []
    for str in np_arr:
        padded = padding(str)
        padded_arr.append(padded)
    return torch.tensor(padded_arr, dtype=torch.uint8)
    
def equality_query_plain_encoding(np_arr, operand_str):
    # TableScanQuery: where np_arr = operand_str
    mask = torch.eq(np_arr, padding(operand_str))
    output = torch.masked_select(np_arr, mask)
    return output

def do_dictionary_encoding(np_arr):
    dict_tensor = TensorDict(np_arr)
    # np_arr -> 1d tensor of int
    encoded_tensor = get_dict_encoded(np_arr, dict_tensor)
    return dict_tensor, encoded_tensor

def equality_query_dictionary_encoding(dict_tensor, encoded_tensor, operand_str):
    # TableScanQuery: where np_arr = operand_str
    # str -> int
    int_scala_value = dict_tensor.lookup(operand_str)
    mask = torch.eq(encoded_tensor, int_scala_value)
    output = torch.masked_select(encoded_tensor, mask)
    return output