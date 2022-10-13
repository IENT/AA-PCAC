import os
import numpy as np
import rlgr
import zlib

def code_YUV(A_quant_idx, N, bitstream_directory=''):
    """ Entropy encode the indices of the quantized components of
    a signal separately.
    
    Args:
        A_quant_idx (3 x N np.array): indices of the quantized components
        N (int): number of symbols
        bitstream_directory (str): directory to store the bitstream files
    
    Returns:
        int: number of bits needed
    
    """
    
    # Code Y, U, V separately 
    numbits_Y = encode_rlgr(A_quant_idx[:, 0], os.path.join(bitstream_directory, 'bitstream_Y.bin'))
    numbits_U = encode_rlgr(A_quant_idx[:, 1], os.path.join(bitstream_directory, 'bitstream_U.bin'))
    numbits_V = encode_rlgr(A_quant_idx[:, 2], os.path.join(bitstream_directory, 'bitstream_V.bin'))

    # Bit count
    bs_size = numbits_Y + numbits_U + numbits_V
    
    return bs_size

def decode_YUV(N, bitstream_directory=''):
    """ Entropy decode the indices of the quantized components of
    a signal separately.

    Args:
        N (int): number of symbols to decode
        bitstream_directory (str): directory where the bitstream files are stored
    
    Returns:
        N x 3 np.array: indices of the quantized components
    """
    Ahat_quant_idx = np.zeros((N, 3))
    Ahat_quant_idx[:, 0] = decode_rlgr(os.path.join(bitstream_directory, 'bitstream_Y.bin'), N=N)
    Ahat_quant_idx[:, 1] = decode_rlgr(os.path.join(bitstream_directory, 'bitstream_U.bin'), N=N)
    Ahat_quant_idx[:, 2] = decode_rlgr(os.path.join(bitstream_directory, 'bitstream_V.bin'), N=N)

    return Ahat_quant_idx

def write_number_to_file(x, filename, bitstream_directory=''):
    """ Writes the int x to the file filename in binary
    
    Args:
        x (int): Integer below 256 that should be binary written to the file
        filename (str): name of the file we write to

    Returns:
        int: number of bits needed (can only be 8 at the moment)
    """
    
    path = os.path.join(bitstream_directory, filename)
    outputFile=open(path, 'wb')
    bytes_count = 1 if (x <= 255) else int(np.ceil(np.ceil(np.log2(x)) / 8))
    outputFile.write(x.to_bytes(bytes_count, byteorder='big'))
    outputFile.close()
  
    return os.path.getsize(path) * 8

def get_number_from_file(filename, bitstream_directory=''):
    """ Reads the file filename and returns the binary number contained
    
    Args:
        filename (str): name of the file we read from

    Returns:
        int: the number contained in the file
    """
    path = os.path.join(bitstream_directory, filename)
    inputFile=open(path, 'rb')
    b = inputFile.read()
    
    return int.from_bytes(b, byteorder='big')
    

def encode_rlgr(data, filename='test_rlgr.bin', is_signed=1):
    """ Write integer data to RLGR

    Args:
        data (np.array): integer input data
        filename (str, optional): file name. Defaults to 'test_rlgr.bin'.

    Returns:
        _type_: _description_
    """

    # Brutal: delete file without warning
    if os.path.isfile(filename):
        os.remove(filename)

    # Open encoder
    do_write = 1
    enc = rlgr.file(filename, do_write)

    # Write data
    enc.rlgrWrite(data, is_signed)

    # Close encoder
    enc.close()

    # Get bits (os.path.getsize returns number of bytes)
    numbits = os.path.getsize(filename) * 8

    return numbits


def decode_rlgr(filename, N, delete_file=False, is_signed=1):
    """Read integer data from RLGR file

    Args:
        filename (str): filename
        N (int): number of integers to decode
        delete_file (bool, optional): delete file after reading.
            Defaults to False.

    Returns:
        _type_: _description_
    """

    # Open decoder
    do_write = 0
    dec = rlgr.file(filename, do_write)

    # Read data from file
    data = np.array(
        dec.rlgrRead(N, is_signed)
    )

    # Close decoder
    dec.close()

    if delete_file:
        os.remove(filename)

    return data


def encode_zlib(x):
    data = zlib.compress(x)
    numbits = len(data) * 8

    return numbits, data


def decode_zlib(data):
    x = zlib.decompress(data)
    
    return x
