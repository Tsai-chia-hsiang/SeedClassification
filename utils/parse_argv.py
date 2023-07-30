import sys
from .dictutils import combine_key_value_list


def parsing_argv()->dict:
    argv = sys.argv
    l = len(argv)
    if  l < 3:
        return {}
    elif l%2 != 1:
        raise ValueError("Format is not right ! it should be like \n \
                         k0 v0 k1 v0 ..."
            )
    
    return combine_key_value_list(argv[1::2], argv[2::2])