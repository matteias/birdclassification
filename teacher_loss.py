import matplotlib.pyplot as plt
import numpy as np

one = [19.62730598449707,
13.198439598083496,
11.133454322814941,
8.381712913513184,
7.495497703552246,
6.609224796295166,
5.748657703399658,
5.210841178894043,
4.890148639678955,
4.7069926261901855,
]

ten = [
2.4201862812042236,
1.281418800354004,
0.9158665537834167,
0.7676378488540649,
0.7110418677330017,
0.6899232268333435,
0.6820194721221924,
0.6734378933906555,
0.6660585999488831,
0.6639185547828674,
]

twoFive = [0.8488777279853821,
0.5855866074562073,
0.5203170776367188,
0.4179116487503052,
0.35794323682785034,
0.41503503918647766,
0.7592496275901794,
0.6751483678817749,
0.5015161633491516,
0.46302729845046997]

fiveO = [0.5272625684738159,
0.38950178027153015,
0.45240461826324463,
0.3734450340270996,
0.4353436827659607,
0.5554015636444092,
0.4109984338283539,
0.4260435700416565,
0.3992200195789337,
0.3323514759540558]


x = np.arange(1, 11)
plt.plot(x, one)
plt.plot(x, ten)
plt.plot(x, twoFive)
plt.plot(x, fiveO)
plt.yscale('log')
plt.legend(["1%", "10%", "25%", "50%"])

plt.xlabel('Epoch')  # Add an x-label to the axes.
plt.ylabel('Loss')  # Add a y-label to the axes.
plt.show()