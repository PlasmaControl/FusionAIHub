    # divide the data into subcategory
def data_division(self, input_file, input_suffix):
    if input_suffix in multi_level:
        input_multi_level = {}
        for key in file_keys[input_suffix].keys():
            keys_of_this_category = file_keys[input_suffix][key]
            input_multi_level[key] = {key_i: input_file[key_i]
                                        for key_i in keys_of_this_category}
    else:
        input_multi_level = {input_suffix: input_file}
    return input_multi_level