from streamlit.ReportThread import get_report_ctx
from functools import cache
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from PIL import Image

MODEL = "nlpconnect/vit-gpt2-image-captioning"


class Prediction:
    @staticmethod
    @cache
    @st.cache(allow_output_mutation=True)
    def get_model():
        model = VisionEncoderDecoderModel.from_pretrained(MODEL)
        feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        nlp = pipeline("image-to-text", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

    @staticmethod
    def get_prediction(image: Image):
        if not image: return
        image_to_text = Prediction.get_model()
        return image_to_text(image)
