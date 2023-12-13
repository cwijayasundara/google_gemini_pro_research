import os
import vertexai
import ssl
import urllib.request

from dotenv import load_dotenv

from IPython.display import Markdown, display
from vertexai.preview.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Image,
    Part,
)

load_dotenv()

context = ssl._create_unverified_context()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-pro-vision")

