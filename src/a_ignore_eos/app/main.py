from fastapi import FastAPI

from .routers.chat import router as chat_router


def create_app() -> FastAPI:
    app = FastAPI(title="llama-custom-api (Pattern A: ignore_eos)", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(chat_router)
    return app


app = create_app()

