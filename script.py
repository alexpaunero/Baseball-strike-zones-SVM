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

# Task 10 and 14
classifier = SVC(kernel='rbf', gamma = 100, C = 100)

# Task 11
classifier.fit(training_set[['plate_x','plate_z']], training_set.type)

# Task 12
draw_boundary(ax, classifier)

# Task 13
print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set.type))

# Task 15
top_score = 0
for gamma in range(1, 20):
  for c in range(1, 20):
    classifier = SVC(kernel='rbf', gamma = gamma, C = c)
    classifier.fit(training_set[['plate_x','plate_z']], training_set.type)
    score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set.type)
    if score > top_score:
      top_score = score
      top_gamma = gamma
      top_c = c

print(top_score) 
print(top_gamma) 
print(top_c) 

# Task 16
def strikezone(player):

  player['type'] = player['type'].map({'S': 1, 'B': 0})

  print(player['type'])

  print(player['plate_x'])

  player = player.dropna(subset = ['type', 'plate_x', 'plate_z'])

  plt.scatter(player.plate_x, player.plate_z, c = player.type, cmap = plt.cm.coolwarm, alpha = 0.25)
  plt.show()

  training_set, validation_set = train_test_split(player, random_state = 1)

  classifier = SVC(kernel='rbf', gamma = 1, C = 3)

  classifier.fit(training_set[['plate_x','plate_z']], training_set.type)

  draw_boundary(ax, classifier)
  ax.set_ylim(-2, 6)
  ax.set_xlim(-3, 3)
  plt.show()

strikezone(jose_altuve)
strikezone(david_ortiz)
