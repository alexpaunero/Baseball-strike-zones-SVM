import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# Task 1
#print(aaron_judge.columns)

# Task 2
print(aaron_judge.description.unique())

# Task 3
print(aaron_judge.type.unique())

# Task 4
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})

# Task 5
print(aaron_judge['type'])

# Task 6
print(aaron_judge['plate_x'])

# Task 7
aaron_judge = aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])
