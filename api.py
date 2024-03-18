from fastapi import FastAPI, UploadFile, File
from service import extract_and_parse_data, getIndicatorData, generate_pdf

app = FastAPI()


@app.post("/esg/benchmark/upload/{entityName}", status_code=200)
async def pdf_report(entityName: str, documentUpload: UploadFile = File(...)):
    return extract_and_parse_data(entityName, documentUpload.file, documentUpload.filename)

@app.post("/esg/benchmark/upload/{entityName}/{esgType}/{esgIndicator}", status_code=200)
async def pdf_report(entityName: str, esgType: str, esgIndicator: str, documentUpload: UploadFile = File(...)):
    return getIndicatorData(entityName, esgType, esgIndicator, documentUpload)

@app.get("/esg/benchmark/keepalive", status_code=200)
def keep_alive():
    return {
        "status": "200",
        "message": "OK"
    }

@app.post("/esg/benchmark/pdf-report/{entityName}", status_code=200)
async def pdf_report(entityName: str, documentUpload: UploadFile = File(...)):
    return generate_pdf(entityName, documentUpload.file, documentUpload.filename)



