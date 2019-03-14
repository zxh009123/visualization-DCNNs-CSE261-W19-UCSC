import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

train_loss = [5.2863, 5.2151, 5.1291, 5.0407, 4.9462, 4.8633, 4.7736, 4.6866, 4.5953, 4.5241, 4.4502, 4.3769, 4.2955, 4.2277, 4.1582, 4.0848, 4.0231, 3.9469, 3.8856, 3.8237, 3.7702, 3.7233, 3.6685, 3.6200, 3.5894, 3.5465, 3.5724, 3.4376, 3.3819, 3.3683, 3.3278, 3.3086, 3.2612, 3.2546, 3.2199, 3.1971, 3.1813, 3.1397, 3.1107, 3.0937, 3.0827, 3.0599, 3.0264, 3.0219, 2.9935, 2.9921, 2.9612, 2.9421, 2.9286, 2.9143, 2.8862, 2.8870, 2.8724, 2.8566, 2.8376, 2.8387, 2.8172, 2.8096, 2.8078, 2.7959, 2.7771, 2.7668, 2.7618, 2.7621, 2.7534, 2.7297, 2.7344, 2.7386, 2.7253, 2.7299, 2.7204, 2.7173, 2.7142, 2.7174, 2.7073, 2.7172, 2.7079, 2.7148, 2.7113]

val_loss = [5.2407, 5.1483, 5.0358, 4.9102, 4.7682, 4.6457, 4.5119, 4.3821, 4.2709, 4.1497, 4.0326, 3.9357, 3.8292, 3.7608, 3.6371, 3.5580, 3.4601, 3.3721, 3.3135, 3.2406, 3.1753, 3.1256, 3.0729, 3.0113, 2.9934, 2.9482, 2.8837, 2.8484, 2.8201, 2.7913, 2.7706, 2.7629, 2.7515, 2.7203, 2.7268, 2.7183, 2.6974, 2.6947, 2.6740, 2.6736, 2.6701, 2.6552, 2.6605, 2.6535, 2.6331, 2.6479, 2.6285, 2.6657, 2.6334, 2.6333, 2.6434, 2.6414, 2.6483, 2.6304, 2.6252, 2.6470, 2.6500, 2.6398, 2.6476, 2.6437, 2.6461, 2.6422, 2.6429, 2.6372, 2.6385, 2.6324, 2.6422, 2.6258, 2.6504, 2.6346, 2.6356, 2.6351, 2.6319, 2.6323, 2.6351, 2.6365, 2.6359, 2.6424, 2.6383]

train_acc = [0.0165, 0.0462, 0.0704, 0.0897, 0.1072, 0.1207, 0.1360, 0.1502, 0.1674, 0.1789, 0.1904, 0.2023, 0.2185, 0.2305, 0.2434, 0.2584, 0.2695, 0.2836, 0.2948, 0.3060, 0.3146, 0.3261, 0.3347, 0.3413, 0.3488, 0.3549, 0.3846, 0.3766, 0.3865, 0.3895, 0.3968, 0.4001, 0.4096, 0.4122, 0.4184, 0.4229, 0.4258, 0.4323, 0.4386, 0.4425, 0.4451, 0.4479, 0.4553, 0.4570, 0.4618, 0.4631, 0.4682, 0.4719, 0.4759, 0.4779, 0.4835, 0.4830, 0.4873, 0.4903, 0.4936, 0.4932, 0.4981, 0.4992, 0.4997, 0.5018, 0.5060, 0.5068, 0.5095, 0.5093, 0.5110, 0.5151, 0.5149, 0.5137, 0.5162, 0.5168, 0.5172, 0.5170, 0.5196, 0.5191, 0.5202, 0.5189, 0.5196, 0.5186, 0.5194]

val_acc = [0.0486, 0.0934, 0.1198, 0.1464, 0.1710, 0.1888, 0.2088, 0.2264, 0.2448, 0.2632, 0.2858, 0.2992, 0.3216, 0.3418, 0.3572, 0.3672, 0.3920, 0.4004, 0.4104, 0.4196, 0.4220, 0.4346, 0.4424, 0.4474, 0.4538, 0.4590, 0.4702, 0.4728, 0.4852, 0.4924, 0.4956, 0.4970, 0.4948, 0.5004, 0.5062, 0.5014, 0.5080,  0.5142, 0.5132, 0.5133, 0.5132, 0.5152, 0.5228, 0.5180, 0.5200, 0.5212, 0.5186, 0.5208, 0.5258, 0.5228, 0.5256, 0.5272, 0.5296, 0.5276, 0.5304, 0.5314, 0.5320, 0.5316, 0.5342, 0.5318, 0.5282, 0.5326, 0.5350, 0.5364, 0.5370, 0.5370, 0.5368, 0.5366, 0.5316, 0.5378, 0.5358, 0.5364, 0.5350, 0.5366, 0.5330, 0.5350, 0.5346, 0.5334, 0.5360]



# 0.3413, 0.3488, 0.3549, 0.4873, 0.3766, 0.3865, 0.3895, 0.3968, 0.4001, 0.4096, 0.4122, 0.4184, 0.4229, 0.4258, 0.4323, 0.4386, 0.4425, 0.4451, 0.4479, 0.4553, 0.4570, 0.4618, 0.4631, 0.4682, 0.4719, 0.4759, 0.4779, 0.4835, 0.4830, 0.4873, 0.4903, 0.4936, 0.4932, 0.4981, 0.4992, 0.4997, 0.5018, 0.5060, 0.5068, 0.5095, 0.5093, 0.5110, 0.5151, 0.5149, 0.5137, 0.5162, 0.5168, 0.5172, 0.5170, 0.5196, 0.5191, 0.5202, 0.5189, 0.5196, 0.5186, 0.5194]
# 3.0113, 2.9934, 2.9482, 2.8837, 2.8484, 2.8201, 2.7913, 2.7706, 2.7629, 2.7515, 2.7203, 2.7268, 2.7183, 2.6974, 2.6947, 2.6740, 2.6736, 2.6701, 2.6552, 2.6605, 2.6535, 2.6331, 2.6479, 2.6285, 2.6657, 2.6334, 2.6333, 2.6434, 2.6414, 2.6483, 2.6304, 2.6252, 2.6470, 2.6500, 2.6398, 2.6476, 2.6437, 2.6461, 2.6422, 2.6429, 2.6372, 2.6385, 2.6324, 2.6422, 2.6258, 2.6504, 2.6346, 2.6356, 2.6351, 2.6319, 2.6323, 2.6351, 2.6365, 2.6359, 2.6424, 2.6383
x = [x for x in range(len(train_loss))]

train_loss = np.array(train_loss)
train_acc = np.array(train_acc)*100.0
val_loss = np.array(val_loss)
val_acc = np.array(val_acc)*100.0
x = np.array(x)

fig, ax = plt.subplots()
ax.plot(x, train_acc, '--', label = 'train acc')
ax.plot(x, val_acc, '--', label = 'val acc')

ax2 = ax.twinx()
ax2.plot(x, train_loss, label = 'train loss')
ax2.plot(x, val_loss, label = 'val loss')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
ax.set(xlabel='epochs', ylabel='Accuracy (%)',title='Acuracy and loss of training and testing set')
ax2.set(ylabel = 'Loss')
plt.show()