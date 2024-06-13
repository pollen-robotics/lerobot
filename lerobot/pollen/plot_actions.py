import pickle

import matplotlib.pyplot as plt

actions = pickle.load(open("actions.pkl", "rb"))

# for action in actions:
#     print(action)
#     exit()

plt.plot(actions)
plt.show()
