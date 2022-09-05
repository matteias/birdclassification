import matplotlib.pyplot as plt
import numpy as np

one = [0.16050000488758087,
0.15850000083446503,
0.17000000178813934,
0.15299999713897705,
0.15800000727176666,
0.1665000021457672,
0.15649999678134918,
0.1615000069141388,
0.15649999678134918,
0.16500000655651093
]

ten = [
0.8180000185966492,
0.8109999895095825,
0.8199999928474426,
0.8149999976158142,
0.8169999718666077,
0.8339999914169312,
0.8240000009536743,
0.8379999995231628,
0.8349999785423279,
0.8324999809265137
]

twoFive = [0.8650000095367432,
0.8554999828338623,
0.8604999780654907,
0.8669999837875366,
0.8675000071525574,
0.875,
0.8859999775886536,
0.8870000243186951,
0.8884999752044678,
0.8855000138282776
]


fiveO = [0.8790000081062317,
0.9045000076293945,
0.9024999737739563,
0.9014999866485596,
0.9085000157356262,
0.9190000295639038,
0.9125000238418579,
0.9200000166893005,
0.9229999780654907,
0.9259999990463257
]


x = np.arange(1, 11)
plt.plot(x, one)
plt.plot(x, ten)
plt.plot(x, twoFive)
plt.plot(x, fiveO)
#plt.yscale('log')
plt.legend(["1%", "10%", "25%", "50%"])

plt.xlabel('Epoch')  # Add an x-label to the axes.
plt.ylabel('Accuracy')  # Add a y-label to the axes.
plt.show()