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

# Task 8
plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c = aaron_judge.type, cmap = plt.cm.coolwarm, alpha = 0.25)
plt.show()

# Task 9
training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

# Task 10
classifier = SVM(kernel='rbf')

# Task 11
classifier.fit(training_set[['plate_x','plate_z']], training_set.type)

# Task 12
draw_boundary(ax, classifier)
