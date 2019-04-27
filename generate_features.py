import numpy as np
import torch

#data_dir = 'CUB_200_2011/images/001.Black_footed_Albatross/'

names = []
with open("CUB_200_2011/images.txt") as f:
    image_list = f.readlines()
    for line in image_list:
        items = line.split(" ")
        names.append(items[1])

indexes = []
a = sorted(names)
for item in a:
    indexes.append(names.index(item))

info = np.loadtxt("CUB_200_2011/attributes/image_attribute_labels.txt")
attribute_values = []
certainty_values = []
for i in range(len(indexes)):
    attribute_values.append(info[i * 312:(i + 1) * 312, 2])
    certainty_values.append(info[i * 312:(i + 1) * 312, 3])

attribute_values = np.array(attribute_values)
certainty_values = np.array(certainty_values)

attribute_values = attribute_values[indexes]
certainty_values = certainty_values[indexes]
print(attribute_values.shape)
print(certainty_values.shape)
np.save("CUB_200_2011/attribute_values.npy", attribute_values)
np.save("CUB_200_2011/certainty_values.npy", certainty_values)

