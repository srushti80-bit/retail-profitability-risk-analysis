import numpy as np

def preprocess_input(Sales, Discount, Quantity, Category, Region, le1, le2):

    cat_enc = le1.transform([Category])[0]
    reg_enc = le2.transform([Region])[0]

    input_data = np.array([[Sales, Discount, Quantity, cat_enc, reg_enc]], dtype=float)

    return input_data


def get_prediction(model, input_data):

    return model.predict(input_data)[0]


def get_probability(model, input_data):

    return model.predict_proba(input_data)[0][1]