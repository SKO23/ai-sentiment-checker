 import gradio as gr
import spaces
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

@spaces.GPU
def analyse_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text."

    result = classifier(text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)

    return f"Prediction: {label}\nConfidence: {score}%"

demo = gr.Interface(
    fn=analyse_sentiment,
    inputs=gr.Textbox(label="Enter a sentence"),
    outputs=gr.Textbox(label="AI Result"),
    title="AI Sentiment Checker",
    description="A simple AI app that detects whether text sounds positive or negative."
)

demo.launch()