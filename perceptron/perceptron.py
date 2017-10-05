import numpy as np
import csv

file_name = 'data/pokemon.csv'
attributes_columns = [5, 6, 7, 8, 9, 10]
types_column = [2, 3]

types_map = {}
percentage_to_train = 0.7  # 70%

learning_rate = 0.1
epoch = 1000


def get_model():
    X, Y = [], []
    types_current_index = 0

    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        raw_data = list(reader)

    # normalize X
    max = 0
    for data_index in range(1, len(raw_data)):
        data = raw_data[data_index]
        for attributes_column_index in attributes_columns:
            if max < float(data[attributes_column_index]):
                max = float(data[attributes_column_index])

    # create X
    for data_index in range(1, len(raw_data)):
        data = raw_data[data_index]
        attributes = []
        for attributes_column_index in attributes_columns:
            attributes.append(float(data[attributes_column_index]) / max)
        X.append(attributes)

        for types_column_index in types_column:
            current_type = data[types_column_index]
            if current_type is not '' and current_type not in types_map:
                types_map[current_type] = types_current_index
                types_current_index += 1

    # create Y
    for data_index in range(1, len(raw_data)):
        data = raw_data[data_index]
        types_array = [0] * len(types_map)
        for types_column_index in types_column:
            current_type = data[types_column_index]
            if current_type is not '':
                types_array[types_map[current_type]] = 1

        normalized_array = [i / sum(types_array) for i in types_array]
        Y.append(normalized_array)

    number_of_training_data = int(len(X) * percentage_to_train)
    X_train = X[:number_of_training_data]
    Y_train = Y[:number_of_training_data]
    X_validate = X[number_of_training_data:]
    Y_validate = Y[number_of_training_data:]

    return X_train, Y_train, X_validate, Y_validate


def train_model(X_train, Y_train):
    weigths = np.random.rand(len(X_train[0]), len(Y_train[0]))  # 5x15
    for i in range(0, epoch):
        for data_index in range(0, len(X_train)):
            X = np.array([X_train[data_index]])
            Y = np.array(Y_train[data_index])

            result = np.dot(weigths.T, X.T).flatten()
            error = np.array([Y - result])
            weigths += learning_rate * np.dot(X.T, error)

    return weigths


def validate_model(X_validate, Y_validate, weights):
    correct, incorrect = 0, 0
    for data_index in range(0, len(X_validate)):
        X = np.array([X_validate[data_index]])

        result = np.dot(weights.T, X.T).flatten()
        index_predicted = np.argmax(result)
        if Y_validate[data_index][index_predicted] > 0:
            correct += 1
        else:
            incorrect += 1

    print('Correct: ', correct)
    print('Incorrect: ', incorrect)


def main():
    X_train, Y_train, X_validate, Y_validate = get_model()
    weights = train_model(X_train, Y_train)
    validate_model(X_validate, Y_validate, weights)


if __name__ == "__main__":
    main()
