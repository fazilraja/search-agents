from crewai import Agent, Task, Crew, Process
from litellm import completion
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List, Dict, Optional, Union
import json
import os
from news_schemas import NewsArticle, NewsSource, NewsQuote, NewsEntity
from datetime import datetime

# Initialize Azure OpenAI configuration
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2023-05-15"
os.environ["OTEL_SDK_DISABLED"] = "true"

class LiteLLMWrapper:
    def __init__(self, model_name="gpt-35-turbo"):
        self.model_name = model_name

    def __call__(self, messages, **kwargs):
        try:
            response = completion(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message
        except Exception as e:
            print(f"Error calling LiteLLM: {str(e)}")
            raise

# Initialize tools and LLM
llm = LiteLLMWrapper(model_name="azure/gpt-35-turbo")
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define specialized news agents
news_finder = Agent(
    role='News Finder',
    goal='Find relevant and reliable news articles on specific topics',
    backstory="""You are an expert at finding credible news sources and articles.
    You understand news reliability, bias, and can identify authoritative sources.""",
    tools=[search_tool, scrape_tool],
    llm=llm,
)

content_analyzer = Agent(
    role='Content Analyzer',
    goal='Analyze news articles for key information and patterns',
    backstory="""You are an expert at analyzing news content, extracting quotes,
    identifying key entities, and determining sentiment and topics.""",
    tools=[scrape_tool],
    llm=llm,
)

news_formatter = Agent(
    role='News Formatter',
    goal='Format news analysis into structured data',
    backstory="""You are a specialist in organizing news analysis into clean,
    standardized formats following specific schemas.""",
    llm=llm,
)

def create_news_analysis_tasks(topic: str, sources: Optional[List[str]] = None) -> List[Task]:
    """Create a comprehensive set of news analysis tasks"""
    tasks = [
        Task(
            description=f"""
                Find the most relevant and reliable news articles about {topic}.
                Requirements:
                - Focus on articles from the past week
                - Prioritize reputable news sources
                - Consider source diversity
                - Evaluate article credibility
                - Collect at least 3 different perspectives
                - Include source URLs in response
                {f'Preferred sources: {", ".join(sources)}' if sources else ''}
            """,
            agent=news_finder,
            expected_output="List of relevant articles with URLs and initial credibility assessment"
        ),
        Task(
            description="""
                Perform deep analysis of the found articles:
                - Extract key quotes and statements
                - Identify main entities and their roles
                - Analyze sentiment and bias
                - Compare different perspectives
                - Identify potential misinformation
                - Extract supporting evidence
            """,
            agent=content_analyzer,
            expected_output="Detailed analysis of each article's content and credibility"
        ),
        Task(
            description="""
                Format and structure the analyzed content:
                - Follow the NewsArticle schema exactly
                - Include all metadata and source information
                - Structure quotes and entities properly
                - Calculate aggregate metrics
                - Highlight key findings
                - Flag potential issues or biases
            """,
            agent=news_formatter,
            expected_output="Structured JSON data following the NewsArticle schema",
            output_file="news_analysis.json"
        )
    ]
    return tasks

def process_news_topic(
    topic: str,
    sources: Optional[List[str]] = None,
    save_results: bool = True
) -> Dict:
    """Process and analyze news articles on a specific topic"""
    try:
        crew = Crew(
            agents=[news_finder, content_analyzer, news_formatter],
            tasks=create_news_analysis_tasks(topic, sources),
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        if save_results:
            try:
                with open('news_analysis.json', 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                    
                # Save additional metadata
                with open('news_analysis_meta.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        "topic": topic,
                        "sources": sources,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": "success"
                    }, f, indent=2)
                    
                return analysis_data
            except FileNotFoundError:
                return json.loads(result)
        else:
            return json.loads(result)
            
    except Exception as e:
        error_data = {
            "error": str(e),
            "topic": topic,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "failed"
        }
        
        if save_results:
            with open('news_analysis_error.json', 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2)
        
        return error_data

if __name__ == "__main__":
    # Example usage
    result = process_news_topic(
        topic="artificial intelligence regulation",
        sources=["Reuters", "Associated Press", "Bloomberg"],
        save_results=True
    )
    print(json.dumps(result, indent=2)) 