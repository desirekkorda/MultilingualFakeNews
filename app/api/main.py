from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(

    title="Multilingual Fake News Detection API",

    version="1.1.0",

    description=(
        "REST API for multilingual fake news detection "
        "powered by XLM-RoBERTa."
    ),

    contact={
        "name": "Desire K. Korda",
        "url": "https://github.com/desirekkorda",
    },

    license_info={
        "name": "MIT License"
    }
)


@app.get("/")
def root():

    return {
        "message": "Welcome to the Multilingual Fake News Detection API",
        "documentation": "/docs",
        "api_version": "1.1.0"
    }


app.include_router(router)