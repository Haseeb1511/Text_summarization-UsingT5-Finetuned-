from transformers import pipeline,T5Tokenizer,T5ForConditionalGeneration
import streamlit as st
from pre_processing import pre_processing


tokenizer_path ="./tokenizer"
model_path = "./model"

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

st.title("Text Summarization")

user_input = st.text_area("Enter passage to summarize below :")

if st.button("Generate"):
    pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)
    pre_processing = pre_processing(user_input)
    summary = pipeline(user_input,max_length=150,do_sample=False)
    result = summary[0]
    st.success(result["summary_text"])







# "Artificial Intelligence (AI) has been transforming various sectors, from healthcare to finance. With advancements in machine learning algorithms and the development of powerful computational resources, AI has become more capable and accessible. In healthcare, AI is being used to diagnose diseases more accurately and quickly, sometimes even outperforming human doctors in tasks like interpreting medical imaging. In the financial sector, AI helps in predicting market trends, automating trading, and enhancing customer service through chatbots and personalized recommendations.Despite the immense potential, there are challenges to be addressed, such as ensuring the ethical use of AI, mitigating biases in algorithms, and ensuring that AI does not replace jobs unnecessarily. As AI continues to evolve, it is crucial that developers, policymakers, and the public collaborate to ensure that AI benefits society as a whole, without compromising fairness, privacy, or employment opportunities."