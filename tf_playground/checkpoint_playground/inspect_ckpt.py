import tensorflow as tf


# def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
#   """Prints tensors in a checkpoint file.
#   If no `tensor_name` is provided, prints the tensor names and shapes
#   in the checkpoint file.
#   If `tensor_name` is provided, prints the content of the tensor.
#   Args:
#     file_name: Name of the checkpoint file.
#     tensor_name: Name of the tensor in the checkpoint file to print.
#     all_tensors: Boolean indicating whether to print all tensors.
#   """
#   try:
#     reader = pywrap_tensorflow.NewCheckpointReader(file_name)
#     if all_tensors:
#       var_to_shape_map = reader.get_variable_to_shape_map()
#       for key in var_to_shape_map:
#         print("tensor_name: ", key)
#         print(reader.get_tensor(key))
#     elif not tensor_name:
#       print(reader.debug_string().decode("utf-8"))
#     else:
#       print("tensor_name: ", tensor_name)
#       print(reader.get_tensor(tensor_name))
#   except Exception as e:  # pylint: disable=broad-except
#     print(str(e))
#     if "corrupted compressed block contents" in str(e):
#       print("It's likely that your checkpoint file has been compressed "
#             "with SNAPPY.")

#tf.print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors)
tf.python.tools.inspect_checkpoint.print_tensors_in_checkpoint_file(file_name='./tmp/mdl_ckpt',tensor_name='',all_tensors='')

#print_tensors_in_checkpoint_file(file_name='./tmp/mdl_ckpt',tensor_name='',all_tensors='')
