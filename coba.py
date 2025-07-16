# # To run this code you need to install the following dependencies:
# # pip install google-genai

# import base64
# import os
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv
# load_dotenv()

# def generate():
#     client = genai.Client(
#         api_key=os.environ.get("GEMINI_API_KEY"),
#     )

#     contents = [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_text(text="""INSERT_INPUT_HERE"""),
#             ],
#         ),
#     ]
#     generate_content_config = types.GenerateContentConfig(
#         thinking_config = types.ThinkingConfig(
#             thinking_budget=-1,
#         ),
#         response_mime_type="text/plain",
#     )

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=contents,
#         config=generate_content_config
#     )
#     print(response)
#     return response

# if __name__ == "__main__":
#     generate()
