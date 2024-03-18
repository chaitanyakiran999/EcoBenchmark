from PyPDF2 import PdfReader
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import PromptTemplate
#
# from llm import analyze_text, get_llm
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.openai import AzureOpenAI

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


esgResponse_template = {
  "entityName": "string",
  "benchmarkDetails": [],
  "metrics": {
    "timeTaken": 0,
    "leveragedModel": "string",
    "f1Score": 0
  }
}

benchmarkDetails_template = {
  "question": "string",
  "esgType": "string",
  "esgIndicators": "string",
  "primaryDetails": "",
  "secondaryDetails": "",
  "citationDetails": "string",
  "pageNumber": 0
}

esgType = {
    # "MSCISustainalytics": "ESGScore",
    "NetZeroTarget": "Environment",
    "InterimEmissionsReductionTarget": "Environment",
    "RenewableElectricityTarget": "Environment",
    "CircularityStratergy": "Environment",
    "DE&ITarget": "Social",
    # "HealthAndSafetyTarget": "Goverance",
    # "SuppluAuditTarget": "Goverance",
    # "SBTi": "Reporting",
    # "CDP": "Reporting",
    # "GRI": "Reporting",
    # "SASB": "Reporting",
    # "TCFD": "Reporting",
    # "Assurance": "Reporting",
}


# esgQuestions = {
#     "MSCISustainalytics": "ESG Risk Rating for MSCI",
#     "NetZeroTarget": "what is net zero target",
#     "InterimEmissionsReductionTarget": "what is the interim emission reduction target",
#     "RenewableElectricityTarget": "what is the Renewable Electricity Target",
#     "CircularityStratergy": "what is the Circularity Stratergy & targets",
#     "DE&ITarget": "what is the Diversity, Equity and Inclusion target",
#     "HealthAndSafetyTarget": "what is the employee health and Safety audit target",
#     "SuppluAuditTarget": "what is supply audit target",
#     "SBTi": "what is the SBTi rating",
#     "CDP": "what is the CDP rating",
#     "GRI": "what is the GRI rating",
#     "SASB": "what is the SASB rating",
#     "TCFD": "what is the TCFD rating",
#     "Assurance": "is the entity focussing on ESG assurance",
# }

esgQuestions = {
    "MSCISustainalytics": "What is the company's ESG Risk Rating for MSCI",
    "NetZeroTarget": "Does the company has any Net Zero Target, Strictly Reply only YES OR NO,If YES Strictly Reply only in NUMBERS in following format: \"PERCENTAGE by FUTURE YEAR\", Do not give any additional information",
    "InterimEmissionsReductionTarget": "Does the company has any Interim emission reduction target, Strictly Reply only YES OR NO,If YES Strictly Reply only in NUMBERS in following format: \"HIGHEST PERCENTAGE by FUTURE YEAR\", Do not give any additional information",
    "RenewableElectricityTarget": "Does the company has any Renewable Electricity Target, Strictly Reply only YES OR NO,If YES Strictly Reply only in NUMBERS in following format: \"HIGHEST PERCENTAGE by FUTURE YEAR\", Do not give any additional information",
    "CircularityStratergy": "Does the company has any Circularity Strategy Target, Strictly Reply only YES OR NO,If YES Strictly Reply only in NUMBERS in following format: \"HIGHEST PERCENTAGE by FUTURE YEAR\", Do not give any additional information",
    "DE&ITarget": "Has the company taken any measures towards Diversity, Equity and Inclusion, Strictly Reply only YES OR NO, Do not give any additional information|||Is there any future target towards Diversity, Equity and Inclusion, Give very accurate answer and Strictly Reply only in NUMBERS in following format: \"HIGHEST PERCENTAGE by FUTURE YEAR\", Do not give any additional information",
    "HealthAndSafetyTarget": "Does the company has any employee health and Safety audit targets, Strictly Reply only YES OR NO, Do not give any additional information",
    "SuppluAuditTarget": "Does the company has any supply chain or logistic audit target, Strictly Reply only YES OR NO",
    "SBTi": "Does the company has any SBTi rating, Strictly Reply only YES OR NO",
    "CDP": "Does the company has any CDP rating, Strictly Reply only YES OR NO",
    "GRI": "Does the company has any GRI rating, Strictly Reply only YES OR NO",
    "SASB": "Does the company has any SASB rating, Strictly Reply only YES OR NO",
    "TCFD": "Does the company has any TCFD rating, Strictly Reply only YES OR NO",
    "Assurance": "Is the company focussing on ESG assurance, Strictly Reply only YES OR NO",
}

conversation = None

# NLP libraries extract relevant text for ESG
def extract_and_parse_data(entityName, pdf_file):
    response = {"esgResponse": []}
    esgEntityResponse = {
        "entityName": entityName,
        "benchmarkDetails": [],
        "metrics": {
            "timeTaken": 0,
            "leveragedModel": "OpenAI",
            "f1Score": 0
        }
    }

    pdf_text = extract_text_from_pdf(pdf_file)
    # get the text chunks
    text_chunks = get_text_chunks(pdf_text)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    print(vectorstore)

    benchmarkDetails = []

    for quest in esgType:
        benchmark = {}
        conversation = get_conversation_chain(
            vectorstore)
        if esgQuestions[quest] == "ESGScore" or esgQuestions[quest] == "Reporting":
            #webscraping logic
            benchmark["question"] = esgQuestions[quest]
            benchmark["esgType"] = esgType[quest]
            benchmark["esgIndicators"] = quest
            benchmark["primaryDetails"] = "primaryDetails"
            benchmark["secondaryDetails"] = "secondaryDetails"
            benchmark["citationDetails"] = "citationReference"
            benchmark["pageNumber"] = 0
        else:
            ai_response = conversation({'question': esgQuestions[quest]})
            benchmark["question"] = esgQuestions[quest]
            benchmark["esgType"] = esgType[quest]
            benchmark["esgIndicators"] = quest
            benchmark["primaryDetails"] = ai_response["answer"]
            benchmark["secondaryDetails"] = "secondaryDetails"
            benchmark["citationDetails"] = "citationReference"
            benchmark["pageNumber"] = 0
        # print(benchmark)
        esgEntityResponse["benchmarkDetails"].append(benchmark)

    print(esgEntityResponse["benchmarkDetails"])
    response["esgResponse"].append(esgEntityResponse)



    # response["esgResponse"].append()
    # print(conversation({'question': "Has the company taken any measures towards Diversity, Equity and Inclusion, Strictly Reply only YES OR NO, Do not give any additional information"}))
    # print(conversation({'question': "Is there any future target towards Diversity, Equity and Inclusion, Give very accurate answer and Strictly Reply only in NUMBERS in following format: \"HIGHEST PERCENTAGE by FUTURE YEAR\", Do not give any additional information"}))


    # text_chunks = extract_text_from_url("https://www.cdp.net/en/responses?queries%5Bname%5D="+ entityName)
    # vectorstore = get_vectorstore(text_chunks)
    # print(text_chunks)
    # result = get_sentences(pdf_text, "Renewable")
    # print(analyze_text(result))

    return response


def query(question):
    print(conversation({'question': question}))

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_text_from_url(url):
    loader = UnstructuredURLLoader(urls=[url])
    data = loader.load()
    # print(data[0])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    text = text_splitter.split_text(data[0].page_content)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key="e77ed5d8f42d470097a79b4e389349d9"
    )
    # embeddings = HuggingFaceInstructEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # combine_docs_chain=chain,
        llm=get_llm(),
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def get_llm():
    return AzureOpenAI(
        deployment_name="EcoBenchmark",
        openai_api_version="2023-05-15",
        model_name="gpt-35-turbo-instruct",
        openai_api_key="e77ed5d8f42d470097a79b4e389349d9",
        azure_endpoint="https://ecobenchmark.openai.azure.com/",
        temperature=0,
        max_tokens=500
    )