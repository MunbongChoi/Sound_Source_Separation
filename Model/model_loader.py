from Model.model1 import Model1
from Model.model2 import Model2

def load_model(model_name, input_dim):
    if model_name == 'model1':
        return Model1(input_dim)
    elif model_name == 'model2':
        return Model2(input_dim)