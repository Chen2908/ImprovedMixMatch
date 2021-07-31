from tensorflow.keras.applications import resnet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


class Baseline:
    def __init__(self, image_size):
        self.image_size = image_size
        self.model = None

    def create_baseline(self, lr, num_classes, metrics, dropout_rate, dense_size):
        base_model = resnet.ResNet101(include_top=False,
                                      input_shape=(self.image_size, self.image_size, 3),
                                      weights='imagenet')

        # freeze all the weights except for the last
        for layer in base_model.layers[:-1]:
            layer.trainable = False

        flat1 = Flatten()(base_model.layers[-1].output)
        class1 = Dense(dense_size, activation='relu')(flat1)
        drop1 = Dropout(dropout_rate)(class1)
        output = Dense(num_classes, activation='softmax')(drop1)
        model = Model(inputs=base_model.inputs, outputs=output)

        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy',
                      metrics=metrics)
        # print(model.summary())
        self.model = model
        return model

    def evaluate(self, test_ds):
        metrics = self.model.evaluate(test_ds)
        results = {'Accuracy': metrics[1], 'TPR': metrics[9],
                   'FPR': metrics[8] / (metrics[5] + metrics[8]), 'Precision': metrics[2], 'AUC': metrics[3],
                   'AUPRC': metrics[4]}
        return metrics[0], results




