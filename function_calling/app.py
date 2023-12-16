import os
import vertexai

from dotenv import load_dotenv

import requests
from vertexai.preview.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]

vertexai.init(project=PROJECT_ID, location=REGION)

model = GenerativeModel("gemini-pro")

get_current_weather_func = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location"
            }
        }
    },
)

weather_tool = Tool(
    function_declarations=[get_current_weather_func],
)

prompt = "What is the weather like in Boston?"

response = model.generate_content(
    prompt,
    generation_config={"temperature": 0},
    tools=[weather_tool],
)
print(response)
print(response.candidates[0].content.parts[0].function_call)

# complex function calling
get_location = FunctionDeclaration(
    name="get_location",
    description="Get latitude and longitude for a given location",
    parameters={
        "type": "object",
        "properties": {
            "poi": {
                "type": "string",
                "description": "Point of interest"
            },
            "street": {
                "type": "string",
                "description": "Street name"
            },
            "city": {
                "type": "string",
                "description": "City name"
            },
            "county": {
                "type": "string",
                "description": "County name"
            },
            "state": {
                "type": "string",
                "description": "State name"
            },
            "country": {
                "type": "string",
                "description": "Country name"
            },
            "postal_code": {
                "type": "string",
                "description": "Postal code"
            },
        },
    },
)

location_tool = Tool(
    function_declarations=[get_location],
)

prompt = """
I want to get the lat/lon coordinates for the following address:
1600 Amphitheatre Pkwy, Mountain View, CA 94043, US
"""

response = model.generate_content(
    prompt,
    generation_config={"temperature": 0},
    tools=[location_tool],
)

print(response.candidates[0].content.parts[0])

x = response.candidates[0].content.parts[0].function_call.args

url = "https://nominatim.openstreetmap.org/search?"
for i in x:
    url += '{}="{}"&'.format(i, x[i])
url += "format=json"

x = requests.get(url)
content = x.json()
print(content)

# Function calling in a chat session

get_product_info_func = FunctionDeclaration(
    name="get_product_sku",
    description="Get the SKU for a product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "Product name"
            }
        }
    },
)

get_store_location_func = FunctionDeclaration(
    name="get_store_location",
    description="Get the location of the closest store",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location"
            }
        }
    },
)

place_order_func = FunctionDeclaration(
    name="place_order",
    description="Place an order",
    parameters={
        "type": "object",
        "properties": {
            "product": {
                "type": "string",
                "description": "Product name"
            },
            "account": {
                "type": "integer",
                "description": "Account number"
            },
            "address": {
                "type": "string",
                "description": "Shipping address"
            }
        }
    },
)

retail_tool = Tool(
    function_declarations=[get_product_info_func,
                           get_store_location_func,
                           place_order_func,
                           ],
)

model = GenerativeModel("gemini-pro",
                        generation_config={"temperature": 0},
                        tools=[retail_tool])
chat = model.start_chat()

prompt = """
Do you have the Pixel 8 Pro in stock?
"""

response = chat.send_message(prompt)
print(response.candidates[0].content.parts[0])

api_response = {"sku": "GA04834-US", "in_stock": "yes"}


response = chat.send_message(
    Part.from_function_response(
        name="get_product_sku",
        response={
            "content": api_response,
        }
    ),
)
print(response.candidates[0].content.parts[0])

prompt = """
Where can I buy one near Mountain View, CA?
"""

response = chat.send_message(prompt)
print(response.candidates[0].content.parts[0])

api_response = {"store": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, US"}

response = chat.send_message(
    Part.from_function_response(
        name="get_store_location",
        response={
            "content":  api_response,
        }
    ),
)
print(response.candidates[0].content.parts[0])

prompt = """
I'd like to order a Pixel 8 Pro and have it shipped to 1155 Borregas Ave, Sunnyvale, CA 94089.
"""

response = chat.send_message(prompt)
print(response.candidates[0].content.parts[0])

api_response = {"payment_status": "paid", "order_number": 12345, "est_arrival": "2 days"}

response = chat.send_message(
    Part.from_function_response(
        name="place_order",
        response={
            "content":  api_response,
        }
    ),
)
print(response.candidates[0].content.parts[0])
