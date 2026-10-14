# Selected student code from the Fall 2022 submission, imported in 2026.
# Fragments require the original notebook variables and library imports.

# Student implementation, submitted cell 32
from sklearn.metrics import pairwise

newDim = pairwise.rbf_kernel(X,np.array([[0, 0]]))

plt.figure(figsize=(12, 7))
ax =plt.axes(projection='3d')

ax.scatter3D(X[:,0],X[:,1],newDim,c=y)

ax.set_xlabel('X axis', fontsize=15)
ax.set_ylabel('Y axis', fontsize=15)
ax.set_zlabel('RBF', fontsize=15)

plt.show()

# Student implementation, submitted cell 37
from sklearn import svm
from sklearn import metrics

clfL = svm.SVC(kernel='linear')
clfL.fit(X_train, y_train)
y_predL = clfL.predict(X_test)

clfR = svm.SVC(kernel='rbf')
clfR.fit(X_train, y_train)
y_predR = clfR.predict(X_test)

print("The accuracy of Gaussian kernel:",metrics.accuracy_score(y_test, y_predR))
print("The accuracy of linear kernel:",metrics.accuracy_score(y_test, y_predL))

# Student implementation, submitted cell 44
simpleModel = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(71,71,3)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(47, activation='softmax')
])

# Student implementation, submitted cell 52
newModel = tf.keras.Sequential([
  
    tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape = (71,71,3), padding='valid'),
    tf.keras.layers.AvgPool2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu',  padding='valid'),
    tf.keras.layers.AvgPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',  padding='valid'),
    tf.keras.layers.AvgPool2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(47, activation='softmax')
])



# Student implementation, submitted cell 64
myModel = tf.keras.Sequential([
  
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu') ,

    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(47, activation='softmax')
])

myModel.compile( loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Student-added transfer head, submitted cell 59; list wrapper added in 2026.
student_transfer_layers = [
    tf.keras.layers.AvgPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(.4),
    tf.keras.layers.Dense(47, activation='softmax')
]
