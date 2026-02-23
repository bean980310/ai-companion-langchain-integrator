import os
import json
import logging
from datetime import datetime
from typing import Any
from abc import abstractmethod

import httpx
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource
)

from pydantic import AnyUrl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-server")
class BaseWeatherServer:
    def __init__(self, api_key: str = "None", base_url: str = "http://api.openweathermap.org/data/2.5", city: str = "New York", state: str = None, country: str = "US"):
        self.api_key = api_key
        self.base_url = base_url
        self.params = self.get_location(city, state, country)

        if not self.api_key:
            raise ValueError("API key is required!")

        self.http_params = {"appid": self.api_key, "units": "metric"}
        self.params.update(self.http_params)

    def get_location(self, city: str, state: str, country: str):
        if country == "US" and state:
            return {"q": f"{city},{state},{country}"}
        else:
            return {"q": f"{city},{country}"}
