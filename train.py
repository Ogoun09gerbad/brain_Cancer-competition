import pandas as pd
from model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train(tr_df, val_df, epochs=10):
    img_size = (299, 299)
    _gen = ImageDataGenerator(rescale=1/255, brightness_range=(0.8, 1.2))
    tr_gen = _gen.flow_from_dataframe(tr_df, x_col='Class Path', y_col='Class', target_size=img_size, batch_size=32)
    val_gen = _gen.flow_from_dataframe(val_df, x_col='Class Path', y_col='Class', target_size=img_size, batch_size=32)
    
    model = build_model()
    hist = model.fit(tr_gen, epochs=epochs, validation_data=val_gen)
    model.save('brain_tumor_model.h5')
    return hist
