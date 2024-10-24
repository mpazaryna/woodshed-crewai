import requests
from crewai_tools import BaseTool


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "A tool to perform searches using DuckDuckGo."

    def _run(self, query: str) -> str:
        # Perform a search using DuckDuckGo
        response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        data = response.json()

        # Extract relevant information from the response
        results = data.get("RelatedTopics", [])
        summaries = [result["Text"] for result in results if "Text" in result]

        return "\n".join(summaries) if summaries else "No results found."


# Example usage
# search_tool = DuckDuckGoSearchTool()
# result = search_tool.run("latest AI news")
# print(result)
