import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def analyse_sentiment(text):
    if not text.strip():
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