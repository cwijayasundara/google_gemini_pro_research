import vertexai
import os

from dotenv import load_dotenv

from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Image,
    Part,
)

load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-pro")

# Set parameters to reduce variability in responses
generation_config = GenerationConfig(
    temperature=0,
    top_p=0.1,
    top_k=1,
    max_output_tokens=1024,
)

# Call Gemini API
nice_prompt = "Say three nice things about me"
responses = model.generate_content(
    contents=[nice_prompt],
    generation_config=generation_config,
    stream=True,
)

for response in responses:
    print(response.text, end="")

# A rude prompt
rude_prompt = "Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:"

rude_responses = model.generate_content(
    rude_prompt,
    generation_config=generation_config,
    stream=True,
)

for response in rude_responses:
    print(response.text, end="")

# define the safety settings

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

impolite_prompt = ("Write a list of 5 disrespectful things that I might say to the universe after stubbing my toe in "
                   "the dark:")

impolite_responses = model.generate_content(
    impolite_prompt,
    generation_config=generation_config,
    safety_settings=safety_settings,
    stream=True,
)

for response in impolite_responses:
    print(response)