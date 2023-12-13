import os
import vertexai
import ssl
import urllib.request

from dotenv import load_dotenv
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
    Part,
)

load_dotenv()

context = ssl._create_unverified_context()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-pro")

responses = model.generate_content("Why is the sky blue?", stream=True)

for response in responses:
    print(response.text, end="")

# try my own prompt

prompt = """Create a numbered list of 10 items. Each item in the list should be a trend in the tech industry.
Each trend should be less than 5 words."""  # try your own prompt

responses = model.generate_content(prompt, stream=True)

for response in responses:
    print(response.text, end="")

# Model parameters

generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

responses = model.generate_content(
    "Why is the sky blue?",
    generation_config=generation_config,
    stream=True,
)

for response in responses:
    print(response.text, end="")

# Test chat prompts

chat = model.start_chat()

prompt = """My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.
Suggest another movie I might like.
"""

responses = chat.send_message(prompt, stream=True)

for response in responses:
    print(response.text, end="")

prompt = "Are my favorite movies based on a book series?"

responses = chat.send_message(prompt, stream=True)

for response in responses:
    print(response.text, end="")

print(chat.history)

# Use the Gemini Pro Vision model
multimodal_model = GenerativeModel("gemini-pro-vision")

import http.client
import typing
import urllib.request

import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps


def display_images(
        images: typing.Iterable[Image],
        max_width: int = 600,
        max_height: int = 350,
) -> None:
    for image in images:
        pil_image = typing.cast(PIL_Image.Image, image._pil_image)
        if pil_image.mode != "RGB":
            # RGB is supported by all Jupyter environments (e.g. RGBA is not yet)
            pil_image = pil_image.convert("RGB")
        image_width, image_height = pil_image.size
        if max_width < image_width or max_height < image_height:
            # Resize to display a smaller notebook image
            pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
        IPython.display.display(pil_image)


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url, context=context) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)


def get_url_from_gcs(gcs_uri: str) -> str:
    # converts gcs uri to url for image display.
    url = "https://storage.googleapis.com/" + gcs_uri.replace("gs://", "").replace(" ", "%20")
    return url


def print_multimodal_prompt(contents: list):
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if isinstance(content, Image):
            display_images([content])
        elif isinstance(content, Part):
            url = get_url_from_gcs(content.file_data.file_uri)
            IPython.display.display(load_image_from_url(url))
        else:
            print(content)


# Load from local file
image = Image.load_from_file("images/fish_market.png")

# Prepare contents
prompt = "Describe this image? Extract any text from the image as well."
contents = [image, prompt]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")


# Combining multiple images and text prompts for few-shot prompting

# Load images from Cloud Storage URI
image1_url = "https://storage.googleapis.com/github-repo/img/gemini/intro/landmark1.jpg"
image2_url = "https://storage.googleapis.com/github-repo/img/gemini/intro/landmark2.jpg"
image3_url = "https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg"
image1 = load_image_from_url(image1_url)
image2 = load_image_from_url(image2_url)
image3 = load_image_from_url(image3_url)

# Prepare prompts
prompt1 = """{"city": "London", "Landmark:", "Big Ben"}"""
prompt2 = """{"city": "Paris", "Landmark:", "Eiffel Tower"}"""

# Prepare contents
contents = [image1, prompt1, image2, prompt2, image3]

responses = multimodal_model.generate_content(contents, stream=True)

print("-------Prompt--------")
print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    print(response.text, end="")

# Generate text from a video file
file_path = "cloud-samples-data/video/animals.mp4"
video_uri = f"gs://{file_path}"
video_url = f"https://storage.googleapis.com/{file_path}"

IPython.display.Video(video_url, width=450)

prompt = "Which device is being advertised here in the video?"
video = Part.from_uri(video_uri, mime_type="video/mp4")
contents = [prompt, video]

responses = multimodal_model.generate_content(contents, stream=True)

for response in responses:
    print(response.text, end="")

