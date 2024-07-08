from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import nest_asyncio

nest_asyncio.apply()

from langchain_community.document_loaders import PlaywrightURLLoader

urls = [
    "https://www.wolframalpha.com/input?i=how+many+elements+in+the+periodic+table",
]

loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])

data = loader.load()

print(data[0])