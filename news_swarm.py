import os
from openai import AzureOpenAI
from typing import List, Dict, Optional, Union
import json
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import re
from dotenv import load_dotenv
from news_schemas import NewsArticle, NewsSource, NewsQuote, NewsEntity


load_dotenv()

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-03-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def search_news(query: str) -> str:
    """Perform news search using Google Custom Search API"""
    try:
        print(f"\n=== Search News Tool ===")
        print(f"Query: {query}")
        
        service = build(
            "customsearch", "v1",
            developerKey=os.getenv("GOOGLE_API_KEY")
        )
        
        cse_id = os.getenv("GOOGLE_CSE_ID")
        
        result = service.cse().list(
            q=query,
            cx=cse_id,
            num=5,
            dateRestrict="d7"  # Last 7 days
        ).execute()
        
        formatted_results = []
        if 'items' in result:
            for item in result['items']:
                formatted_results.append({
                    'title': item['title'],
                    'link': item['link'],
                    'snippet': item['snippet'],
                    'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time')
                })
        
        return json.dumps(formatted_results, indent=2)
        
    except Exception as e:
        return f"Error performing news search: {str(e)}"

def extract_article_content(text: str, url: str = None) -> Dict:
    """Extract article information and analyze content"""
    try:
        messages = [
            {"role": "system", "content": """You are a news analysis specialist. Extract and analyze article content according to this schema:
            {
                "title": string,
                "summary": string,
                "quotes": [{"text": string, "speaker": string, "context": string}],
                "entities": [{"name": string, "type": string, "sentiment": number}],
                "sentiment_score": number (-1 to 1),
                "topics": [string],
                "category": string
            }"""},
            {"role": "user", "content": text}
        ]
        
        response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        article_json = json.loads(response.choices[0].message.content)
        if url:
            article_json["url"] = url
        return article_json
            
    except Exception as e:
        return {
            "error": f"Failed to analyze article: {str(e)}",
            "url": url if url else None
        }

def scrape_news_article(url: str, analyze: bool = True) -> Union[str, Dict]:
    """Enhanced web scraping with optional content analysis"""
    try:
        print(f"\n=== Scrape Article Tool ===")
        print(f"URL: {url}")
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.decompose()
        
        text = soup.get_text()
        
        if analyze:
            result = extract_article_content(text, url)
            print("\nAnalyzed Article Content:")
            print(json.dumps(result, indent=2))
            return result
            
        return text
        
    except Exception as e:
        return f"Error scraping article: {str(e)}"

# Add these classes after the existing functions

class SwarmAgent:
    def __init__(self, name: str, description: str, tools: List[Dict] = None):
        """Initialize a swarm agent with specific capabilities"""
        self.name = name
        self.description = description
        self.tools = tools or []
        self.tool_implementations = {
            "search_news": search_news,
            "scrape_news_article": scrape_news_article
        }
        
        self.system_message = f"""You are {name}. {description}
        You have access to tools for searching news and analyzing articles.
        Use these tools to provide accurate and comprehensive news analysis."""

    def run(self, prompt: str) -> Dict:
        """Execute the agent's task based on the prompt"""
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            
            print('Starting news analysis run')
            response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=messages,
                functions=self.tools if self.tools else None,
                temperature=0.7
            )
            
            # Handle function calling if present
            while response.choices[0].finish_reason == "function_call":
                function_call = response.choices[0].message.function_call
                function_name = function_call.name
                function_args = json.loads(function_call.arguments)
                
                print(f"\n=== Tool Call ===")
                print(f"Function: {function_name}")
                print(f"Arguments: {json.dumps(function_args, indent=2)}")
                
                function_response = self.tool_implementations.get(
                    function_name, 
                    lambda **kwargs: f"Error: Function {function_name} not found"
                )(**function_args)
                
                messages.extend([
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": function_name,
                            "arguments": function_call.arguments
                        }
                    },
                    {
                        "role": "function",
                        "name": function_name,
                        "content": str(function_response)
                    }
                ])
                
                response = client.chat.completions.create(
                    model="gpt-35-turbo",
                    messages=messages,
                    functions=self.tools if self.tools else None,
                    temperature=0.7
                )
            
            return {
                "data": {
                    "output": response.choices[0].message.content
                }
            }
        except Exception as e:
            error_msg = f"Error in agent execution: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

class OrchestratorAgent(SwarmAgent):
    """Coordinates multiple specialized agents for comprehensive news analysis"""
    def __init__(self, sub_agents: Dict[str, SwarmAgent]):
        super().__init__(
            name="News Orchestrator",
            description="I coordinate between specialized agents to perform comprehensive news analysis",
            tools=[
                {
                    "name": "search_news",
                    "description": "Search for news articles",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "scrape_news_article",
                    "description": "Scrape and analyze news article content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "analyze": {"type": "boolean"}
                        },
                        "required": ["url"]
                    }
                }
            ]
        )
        self.sub_agents = sub_agents

    def run(self, prompt: str) -> Dict:
        """Execute orchestrated news analysis workflow"""
        try:
            # Plan the analysis tasks
            planning_messages = [
                {"role": "system", "content": f"""You are a news analysis orchestrator.
                Available agents: {', '.join(self.sub_agents.keys())}.
                Create a plan to analyze news content and return as JSON array with:
                - agent_name: which agent to use
                - sub_task: specific task for the agent"""},
                {"role": "user", "content": prompt}
            ]
            
            planning_response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=planning_messages,
                temperature=0.7
            )
            
            tasks = json.loads(planning_response.choices[0].message.content)
            if not isinstance(tasks, list):
                tasks = [tasks]
            
            # Execute tasks and collect results
            results = []
            for task in tasks:
                agent_name = task.get("agent_name")
                sub_task = task.get("sub_task")
                
                if agent_name in self.sub_agents:
                    result = self.sub_agents[agent_name].run(sub_task)
                    results.append({
                        "agent": agent_name,
                        "task": sub_task,
                        "result": result
                    })
            
            # Synthesize results
            synthesis_messages = [
                {"role": "system", "content": "Synthesize the news analysis results into a final report."},
                {"role": "user", "content": f"Results: {json.dumps(results, indent=2)}"}
            ]
            
            synthesis = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=synthesis_messages,
                temperature=0.7
            )
            
            return {
                "data": {
                    "output": synthesis.choices[0].message.content,
                    "analysis_results": results
                }
            }
            
        except Exception as e:
            error_msg = f"Orchestration error: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

# Add example usage
if __name__ == "__main__":
    # Create specialized agents
    news_searcher = SwarmAgent(
        name="News Searcher",
        description="Specializes in finding relevant news articles from reliable sources",
        tools=[{
            "name": "search_news",
            "description": "Search for news articles",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }]
    )
    
    content_analyzer = SwarmAgent(
        name="Content Analyzer",
        description="Analyzes news article content for key information and patterns",
        tools=[{
            "name": "scrape_news_article",
            "description": "Scrape and analyze news article content",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "analyze": {"type": "boolean"}
                },
                "required": ["url"]
            }
        }]
    )
    
    # Create orchestrator
    orchestrator = OrchestratorAgent({
        "news_searcher": news_searcher,
        "content_analyzer": content_analyzer
    })
    
    # Test the system
    result = orchestrator.run(
        "Analyze recent news about artificial intelligence developments"
    )
    print(json.dumps(result, indent=2))