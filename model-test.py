from model import Model
from model_example_data import CONTEXT

def test_model_notEmpty():
    prediction = Model.get_prediction(image_to_text = Prediction.get_model()) 
    assert prediction['image_to_text'] != ""
