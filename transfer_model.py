import jittor as jt
import torch
import numpy as np
import collections

jittor_data = jt.load('./models/256x256_classifier.pt')
print(type(jittor_data))

# f = open('output.txt', 'w')
# f.write(str(jittor_data) + '\n')

torch_data = torch.load('./models/256x256_classifier.pt', map_location='cpu')
print(type(torch_data))
# f.write(str(torch_data))

dict_jt = collections.OrderedDict()

for layer in torch_data:
    np_arr = torch_data[layer].detach().numpy()
    with jt.enable_grad():
        jt_arr = jt.array(np_arr)
    dict_jt[layer] = jt_arr
    
# f.write(str(dict_jt))
jt.save(dict_jt, './models/256x256_classifier.pkl')