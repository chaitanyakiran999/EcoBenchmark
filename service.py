import re
import datetime

from PyPDF2 import PdfReader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.openai import AzureOpenAI

import requests
from bs4 import BeautifulSoup
import json

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

esgTypes = {
    "MSCISustainalytics": "ESGScore",
    "NetZeroTarget": "Environment",
    "InterimEmissionsReductionTarget": "Environment",
    "RenewableElectricityTarget": "Environment",
    "CircularityStratergy": "Environment",
    "DE&ITarget": "Social",
    "HealthAndSafetyTarget": "Goverance",
    "SuppluAuditTarget": "Goverance",
    "SBTi": "Reporting",
    "CDP": "Reporting",
    "GRI": "Reporting",
    "SASB": "Reporting",
    "TCFD": "Reporting",
    "Assurance": "Reporting",
}

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
    "CDP": "What is the latest CDP rating, Strictly give only the latest Rating",
    "GRI": "Does the company follows Global Reporting Initiative (GRI) standards, Give very accurate answer, Strictly Reply only YES OR NO",
    "SASB": "Does the company follows Sustainability Accounting Standards Board (SASB) standards rating, Give very accurate answer, Strictly Reply only YES OR NO",
    "TCFD": "Does the company follows Task Force on Climate-Related Disclosures (TCFD) standards, Give very accurate answer, Strictly Reply only YES OR NO",
    "Assurance": "Is the company focussing on ESG assurance, Strictly Reply only YES OR NO",
}

conversation = None

def extract_and_parse_data(entityName, pdf_file, filename):
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

    for esgIndicator in esgTypes:
        benchmark = {}

        if esgIndicator != "MSCISustainalytics" or esgIndicator != "SBTi" or esgIndicator != "CDP":
            benchmark = get_data_from_pdf(esgIndicator, esgTypes[esgIndicator], vectorstore, filename)

        if esgIndicator == "MSCISustainalytics":
            benchmark = get_msci_sustainalytics_data(entityName, esgTypes[esgIndicator], esgIndicator)

        if esgIndicator == "SBTi":
            benchmark = get_sbti_data(entityName, esgIndicator)

        if esgIndicator == "CDP":
            benchmark = get_cdp_data(entityName, esgIndicator)

        esgEntityResponse["benchmarkDetails"].append(benchmark)

    print(esgEntityResponse["benchmarkDetails"])
    response["esgResponse"].append(esgEntityResponse)
    return response


def get_data_from_pdf(esgIndicator, esgType, vectorstore, filename):
    benchmark = {}
    conversation = get_conversation_chain(vectorstore)
    ai_response = conversation({'question': esgQuestions[esgIndicator]})
    print(ai_response["answer"])
    primary_details = ""
    secondary_details = ""
    if esgType == "Environment" or esgType == "Social":
        if str(ai_response["answer"]).lower().__contains__("yes"):
            primary_details = "YES"
        else:
            primary_details = "NO"

        percentage_pattern = r'\b\d{2}\b'
        year_pattern = r'\b\d{4}\b'
        percentage = re.findall(percentage_pattern, ai_response["answer"])
        year = re.findall(year_pattern, ai_response["answer"])
        print(percentage)
        print(year)
        if len(percentage) != 0 and len(year) != 0 and int(year[0]) > datetime.datetime.now().year:
            secondary_details = percentage[0] + "% by " + year[0]
    else:
        if str(ai_response["answer"]).lower().__contains__("yes"):
            primary_details = "YES"
        else:
            primary_details = "NO"

    benchmark["question"] = esgQuestions[esgIndicator]
    benchmark["esgType"] = esgType
    benchmark["esgIndicators"] = esgIndicator
    benchmark["primaryDetails"] = primary_details
    benchmark["secondaryDetails"] = secondary_details
    benchmark["citationDetails"] = filename
    benchmark["pageNumber"] = 0
    return benchmark


def getIndicatorData(entityName, esgType, esgIndicator, documentUpload):
    pdf_text = extract_text_from_pdf(documentUpload.file)
    # get the text chunks
    text_chunks = get_text_chunks(pdf_text)
    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    data = []

    if esgIndicator != "MSCISustainalytics" or esgIndicator != "SBTi" or esgIndicator != "CDP":
        data = get_data_from_pdf(esgIndicator, esgType, vectorstore, documentUpload.filename)

    if esgIndicator == "MSCISustainalytics":
        data = get_msci_sustainalytics_data(entityName, esgTypes[esgIndicator], esgIndicator)

    if esgIndicator == "SBTi":
        data = get_sbti_data(entityName, esgIndicator)

    if esgIndicator == "CDP":
        data = get_cdp_data(entityName, esgIndicator)

    esgEntityResponse = {
        "entityName": entityName,
        "benchmarkDetails": data,
        "Metrics": {
            "timeTaken": 0,
            "leveragedModel": "OpenAI",
            "f1Score": 0
        }
    }
    response = {"esgResponse": []}
    response["esgResponse"].append(esgEntityResponse)

    return response

def get_msci_sustainalytics_data(entityName, esgType, esgIndicator):
    benchmark = {}
    rating = getMSCIRating(entityName)
    score = getESGScore(entityName)
    benchmark["question"] = esgQuestions[esgIndicator]
    benchmark["esgType"] = esgType
    benchmark["esgIndicators"] = esgIndicator
    benchmark["primaryDetails"] = rating
    benchmark["secondaryDetails"] = score
    benchmark["citationDetails"] = "citationReference"
    benchmark["pageNumber"] = 0
    return benchmark

def get_sbti_data(entityName, esgIndicator):
    benchmark = {}
    isPresent = "NO"
    if isCompanyPresentSBTi(entityName):
        isPresent = "YES"
    benchmark["question"] = esgQuestions[esgIndicator]
    benchmark["esgType"] = esgTypes[esgIndicator]
    benchmark["esgIndicators"] = esgIndicator
    benchmark["primaryDetails"] = isPresent
    benchmark["secondaryDetails"] = ""
    benchmark["citationDetails"] = "citationReference"
    benchmark["pageNumber"] = 0
    return benchmark


def get_cdp_data(entityName, esgIndicator):
    benchmark = {}
    score = getCDPScore(entityName)
    benchmark["question"] = esgQuestions[esgIndicator]
    benchmark["esgType"] = esgTypes[esgIndicator]
    benchmark["esgIndicators"] = esgIndicator
    benchmark["primaryDetails"] = score
    benchmark["secondaryDetails"] = ""
    benchmark["citationDetails"] = "citationReference"
    benchmark["pageNumber"] = 0
    return benchmark


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
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
    embeddings = OpenAIEmbeddings()
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
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version="2023-05-15",
        model_name="gpt-35-turbo-instruct",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0,
        max_tokens=500
    )


def getESGScore(name):
    url = "https://www.sustainalytics.com/sustapi/companyratings/GetCompanyDropdown"

    name_chunk = name.split()

    payload = {'filter': name_chunk[0],
               'page': '1',
               'pagesize': '1',
               'resourcePackage': 'Sustainalytics'}

    response = requests.request("POST", url, data=payload)

    html = response.text
    parsed_html = BeautifulSoup(html)
    actualName=name
    if parsed_html.find('div', {'class': 'companyName'}) is not None:
        actualName = parsed_html.find('div', {'class': 'companyName'}).span.text
    print(actualName)

    score = "null"
    if actualName is not None:
        code = parsed_html.find('a', attrs={'class': 'search-link'})['data-href']
        print(code)
        url2 = "https://www.sustainalytics.com/esg-rating" + code
        response2 = requests.request("GET", url2)
        score = BeautifulSoup(response2.text).find('div', attrs={'class': 'risk-rating-score'}).span.text
        print(score)

    return score

def getCompanyName(e):
    return e['companyName']


def isCompanyPresentSBTi(name):
    gqlUrl = "https://sciencebasedtargets.org/api"

    payload = json.dumps({
        "query": "\n\tfragment DataEntry on ctaTableEntries_ctaTableEntries_Entry {\n\t\tid\n\t\tlongTermTargetDate\n\t\tlongTermTargetClassification\n\t\tlongTermTargetStatus\n\t\tnearTermTargetClassification\n\t\tnearTermTargetStatus\n\t\tsector\n\t\tsme\n\t\tnetZeroDate\n\t\tnetZeroStatus\n\t\tnetZeroCommitmentStatus\n\t\ttargetDescription\n\t\ttargetClassification\n\t\tba15Status\n\t\tcompanyName\n\t\tdate @formatDateTime(format: \"Y\")\n\t\tnearTermTargetDate\n\t\tregion\n\t\tlocation: cta_location\n\t}\n\n\tquery DataQuery(\n\t\t$orderBy: String\n\t\t$offset: Int\n\t\t$date: [QueryArgument]\n\t\t$sector: [QueryArgument]\n\t\t$sme: [QueryArgument]\n\t\t$region: [QueryArgument]\n\t\t$location: [QueryArgument]\n\t\t$nearTermTargetStatus: [QueryArgument]\n\t\t$nearTermTargetDate: [QueryArgument]\n\t\t$nearTermTargetClassification: [QueryArgument]\n\t\t$longTermTargetStatus: [QueryArgument]\n\t\t$longTermTargetClassification: [QueryArgument]\n\t\t$longTermTargetDate: [QueryArgument]\n\t\t$netZeroStatus: Boolean\n\t\t$netZeroDate: [QueryArgument]\n\t\t$netZeroCommitmentStatus: [QueryArgument]\n\t) {\n\t\tentries(\n\t\t\tlimit: 10000\n\t\t\tsection: \"ctaTableEntries\"\n\t\t\torderBy: $orderBy\n\t\t\toffset: $offset\n\t\t\tdate: $date\n\t\t\tsector: $sector\n\t\t\tsme: $sme\n\t\t\tregion: $region\n\t\t\tcta_location: $location\n\t\t\tnearTermTargetStatus: $nearTermTargetStatus\n\t\t\tnearTermTargetDate: $nearTermTargetDate\n\t\t\tnearTermTargetClassification: $nearTermTargetClassification\n\t\t\tlongTermTargetStatus: $longTermTargetStatus\n\t\t\tlongTermTargetClassification: $longTermTargetClassification\n\t\t\tlongTermTargetDate: $longTermTargetDate\n\t\t\tnetZeroStatus: $netZeroStatus\n\t\t\tnetZeroDate: $netZeroDate\n\t\t\tnetZeroCommitmentStatus: $netZeroCommitmentStatus\n\t\t) {\n\t\t\t...DataEntry\n\t\t}\n\n\t\tkeys: entry(\n\t\t\tsection: \"companiesTakingAction\"\n\t\t\tslug: \"companies-taking-action\"\n\t\t) {\n\t\t\t... on companiesTakingAction_companiesTakingAction_Entry {\n\t\t\t\tkeyIntroduction\n\t\t\t\tctaKey {\n\t\t\t\t\tkeyText\n\t\t\t\t\tkeyColour\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n"
    })

    headers = {
        'Authorization': 'Bearer yQCk41eilBFUbnKcumndiwOZUZoLj7OD',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", gqlUrl, headers=headers, data=payload)
    companies = list(map(getCompanyName, response.json()['data']['entries']))
    isCompanyPresent = name in companies
    print(name)
    print(isCompanyPresent)
    return isCompanyPresent



def getMSCIRating(name):
    #print(name)
    encodedName = name.replace(" ", "-").replace(".", "").replace(",", "").lower()
    #print(encodedName)
    url = "https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool?p_p_id=esgratingsprofile&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=searchEsgRatingsProfiles&p_p_cacheability=cacheLevelPage&_esgratingsprofile_keywords=" + encodedName
    payload = {}
    headers = {
        'User-Agent': ''
    }
    response = requests.request("GET", url, headers=headers, data=payload)
    for row in response.json():
        actualEncodedName = row['encodedTitle']
        if actualEncodedName.startswith(encodedName):
            urlCode = row['url']
            #print(urlCode)
            url2 = "https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool?p_p_id=esgratingsprofile&p_p_lifecycle=2&p_p_state=normal&p_p_mode=view&p_p_resource_id=showEsgRatingsProfile&p_p_cacheability=cacheLevelPage&_esgratingsprofile_issuerId=" + urlCode
            payload2 = {}
            headers2 = {
                'User-Agent': '',
                'Referer': 'https://www.msci.com/our-solutions/esg-investing/esg-ratings-climate-search-tool/issuer/' + actualEncodedName + '/' + urlCode,
            }
            response2 = requests.request("GET", url2, headers=headers2, data=payload2)
            rattingDivClasses = BeautifulSoup(response2.text, features="html.parser").find('div', attrs={'class': 'ratingdata-company-rating'})['class']
            for rattingClass in rattingDivClasses:
                if rattingClass.startswith('esg-rating-circle-'):
                    #print(rattingClass)
                    ratting = rattingClass.replace('esg-rating-circle-', '').upper()
                    print(ratting)
                    return ratting


def getCDPScore(name):
    encodedName = name.replace(".", "").replace(",", "").replace("Inc", "").lower().strip()

    url = "https://www.cdp.net/en/responses?queries%5Bname%5D=" + encodedName
    responseCDP = requests.request("GET", url)
    parsed_html = BeautifulSoup(responseCDP.text, features="html.parser")
    actualNameDiv = parsed_html.find('div', {'class': 'investor-program__score_band_single'})
    if actualNameDiv is None:
        return None
    CDPScore = actualNameDiv.text.replace("\n", "")
    return CDPScore