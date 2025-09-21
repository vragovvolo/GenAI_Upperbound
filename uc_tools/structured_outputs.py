import os
import json
from openai import OpenAI

DATABRICKS_TOKEN = os.environ.get('YOUR_DATABRICKS_TOKEN')
DATABRICKS_BASE_URL = os.environ.get('YOUR_DATABRICKS_BASE_URL')

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=DATABRICKS_BASE_URL
  )

response_format = {
      "type": "json_schema",
      "json_schema": {
        "name": "research_paper_extraction",
        "schema": {
          "type": "object",
          "properties": {
            "title": { "type": "string" },
            "authors": {
              "type": "array",
              "items": { "type": "string" }
            },
            "abstract": { "type": "string" },
            "keywords": {
              "type": "array",
              "items": { "type": "string" }
            }
          },
        },
        "strict": True
      }
    }

messages = [{
        "role": "system",
        "content": "You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given structure."
      },
      {
        "role": "user",
        "content": "..."
      }]

response = client.chat.completions.create(
    model="databricks-gpt-oss-20b",
    messages=messages,
    response_format=response_format
)

print(json.dumps(response.choices[0].message.model_dump()['content'], indent=2))