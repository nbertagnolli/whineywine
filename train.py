import keras
import keras.backend as KK
from keras import models
from keras import layers
from keras.layers import Input, Dense, Flatten, Dropout, LSTM, Lambda
from keras.applications import vgg16

dropout_rate = 0.5
image_shape = (224, 244, 3) # TODO define this!
num_lstm_layers = 4
output_size = 128 # TODO Character output length

def expand_dims(X):
    return KK.expand_dims(X, axis=-1)

def define_model():
    input_tensor = Input(shape=image_shape)
    vgg_conv = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False, input_shape=image_shape)

    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    flattened = Flatten()(vgg_conv.output)
    h = Dense(1024, activation='relu')(flattened)
    h = Dropout(dropout_rate)(h)
    classifier_out = Dense(2, activation='softmax')(h)

    flatten_expanded = Lambda(expand_dims)(flattened)

    lstm_in = flatten_expanded
    lstm_outs = []
    for nidx in range(num_lstm_layers):
        lstm_out = LSTM(32, return_sequences=True)(lstm_in)
    
        slice_layer = Lambda(lambda x: x[:, -1, :])(lstm_out)
        lstm_outs.append(Dropout(dropout_rate)(slice_layer))

        lstm_in = layers.concatenate([lstm_out, flatten_expanded])

    output_layer = layers.concatenate(lstm_outs)
    main_output = Dense(output_size, activation='softmax', name='main_output')(output_layer)

    model = models.Model(inputs=[input_tensor], outputs=[main_output, classifier_out])

    return model

def process_params():
    # Process params
    pass

if __name__ == '__main__':
    model = define_model()
    print(model.summary())
