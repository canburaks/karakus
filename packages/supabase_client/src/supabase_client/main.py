from fastapi import FastAPI

from .client import ClientDep
from .routers import router as auth_router

app = FastAPI(dependencies=[ClientDep])
app.include_router(router=auth_router, prefix="/auth")
