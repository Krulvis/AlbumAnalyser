import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('output_resnext101.csv')
df = df.sample(frac=1).reset_index(drop=True)

df2 = df#df[df.genre.isin(['metal', 'folk', 'jazz'])]
# 10 Genres
n_classes = len(df2.genre.unique())
Y = df2.genre
X = df2.iloc[:, 0:1000]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
labels = list(encoder.inverse_transform(range(0,n_classes)))
# encoded_Y = keras.utils.to_categorical(encoded_Y)
binarized_Y = label_binarize(encoded_Y, classes=range(0,n_classes))
X_train, X_test, Y_train, Y_test = train_test_split(X, binarized_Y, test_size=0.33)

def get_model(genres=10, dropout_1=0.5, output_1=256, activation_1='relu',
              dropout_2=0.5, four=False):
    model = Sequential()
    # Model has an input_dimension of 1000 (all the words)
    model.add(Dense(1000, input_dim=1000, activation='relu'))
    model.add(Dropout(dropout_1))
    model.add(Dense(output_1))
    model.add(Activation(activation_1))
    model.add(Dropout(dropout_2))

    # Optimizing the number of layers
    if four:
        model.add(Dense(100, input_dim=output_1, activation='relu'))
        # Choosing between complete sets of layers
        model.add(Dropout(0.5, activation='linear'))

    # Adding the last dense softmax layer
    model.add(Dense(genres, activation='softmax' if genres > 2 else 'sigmoid'))
    adm = Adam(lr=0.001, beta_1=0.7, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy',
                  optimizer=adm,
                  metrics=['accuracy'])
    return model


k_model = KerasClassifier(build_fn=get_model, verbose=0)
# model = get_model(n_classes if n_classes > 2 else n_classes - 1)


# grid search epochs, batch size
epochs = [1, 10, 20]  # add 50, 100, 150 etc
batch_size = [10, 64, 256, 512]  # add 5, 10, 20, 40, 60, 80, 100 etc
param_grid = dict(epochs=epochs, batch_size=batch_size)

grid = GridSearchCV(estimator=k_model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# result = model.fit(X_train, Y_train,
#           epochs=20,
#           batch_size=128)

# score = model.evaluate(X_test, Y_test, batch_size=128)