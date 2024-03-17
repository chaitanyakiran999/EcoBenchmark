# import PyPDF2
# import streamlit as st
# from langchain_openai import AzureOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from dotenv import load_dotenv
# import os
#
# # Load environment variables
# load_dotenv()
#
# def extract_text_from_pdf(pdf_file):
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page_num in range(len(pdf_reader.pages)):
#         page = pdf_reader.pages[page_num]
#         text += page.extract_text()
#     return text
#
#
#
#
# def analyze_text(text):
#     # Initialize the Azure OpenAI LLM
#     llm = get_llm()
#
#     # Define the prompt template
#     prompt_template = PromptTemplate(
#         input_variables=["text"],
#         template="Analyze the following text and provide ONLY Yes or No response if any steps taken towards Renewable Energy :\n\n{text}",
#     )
#
#     # Create the LLMChain
#     chain = LLMChain(llm=llm, prompt=prompt_template)
#
#     # Run the analysis
#     analysis = chain.invoke(text)
#
#     return analysis['text']
#
# def main():
#     st.title("PDF Parser and Text Analyzer")
#
#     # Create a file uploader for PDF files
#     pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
#
#     if pdf_file is not None:
#         # Parse the PDF file
#         text = extract_text_from_pdf(pdf_file)
#
#         # Analyze the text
#         response = analyze_text("Cricket is a gentlemens game.")
#
#         st.subheader("Text Analysis")
#         st.write(response)
#
#     else:
#         st.write("Please upload a PDF file.")
#
# if __name__ == "__main__":
#     main()