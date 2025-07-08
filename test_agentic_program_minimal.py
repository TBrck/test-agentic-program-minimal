"""
Minimalistic Agentic Program for Data Analysis
=============================================

This program demonstrates how to build an agent that can autonomously decide 
what actions to take based on user input. It's designed for data analysts 
working with advertising/marketing data.

The agent can:
- Filter and analyze datasets
- Answer questions about data insights
- Provide domain knowledge about online advertising
- Generate visualizations and summaries

Architecture:
- Agent Orchestrator: Routes user queries to appropriate tools
- Tools: Modular functions that perform specific tasks
- Context: Maintains conversation state and data context
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Real LLM client using Azure OpenAI (same as backend)
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Import Azure OpenAI configuration from separate file
try:
    from config_azure_openai import (
        AZURE_OPENAI_KEY,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_MODEL
    )
except ImportError:
    raise ImportError(
        "Could not import Azure OpenAI configuration. "
        "Please ensure 'config_azure_openai.py' exists in the same directory with your credentials."
    )

# Set environment variables from config
os.environ['AZURE_OPENAI_KEY'] = AZURE_OPENAI_KEY
os.environ['AZURE_OPENAI_API_VERSION'] = AZURE_OPENAI_API_VERSION
os.environ['AZURE_OPENAI_ENDPOINT'] = AZURE_OPENAI_ENDPOINT
os.environ['AZURE_OPENAI_MODEL'] = AZURE_OPENAI_MODEL

load_dotenv(override=True)

# Initialize real LLM client
llm_client = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_MODEL,
    api_version=AZURE_OPENAI_API_VERSION,
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample advertising data for demonstration with caching
def generate_sample_data() -> pd.DataFrame:
    """
    Generate or load cached sample advertising campaign data.
    Data is cached in temp_results_agentic_minimal folder.
    Delete the folder to force regeneration.
    """
    import os
    
    # Define cache directory and file path
    cache_dir = "temp_results_agentic_minimal"
    cache_file = os.path.join(cache_dir, "sample_advertising_data.csv")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            print(f"📁 Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['date'])
            print(f"✅ Loaded {len(df)} records from cache")
            return df
        except Exception as e:
            print(f"⚠️ Error loading cached data: {e}")
            print("🔄 Generating new data...")
    
    # Generate new data if cache doesn't exist or failed to load
    print(f"🔄 Generating new sample data (cache not found at {cache_file})")
    
    np.random.seed(42)
    n_records = 1000
    
    campaigns = ['Search_Brand', 'Display_Retargeting', 'Social_Video', 'Search_Generic', 'Display_Prospecting']
    channels = ['Google_Ads', 'Facebook', 'LinkedIn', 'YouTube', 'Bing']
    devices = ['Desktop', 'Mobile', 'Tablet']
    
    data = {
        'date': pd.date_range('2024-01-01', periods=n_records, freq='H'),
        'campaign': np.random.choice(campaigns, n_records),
        'channel': np.random.choice(channels, n_records),
        'device': np.random.choice(devices, n_records),
        'impressions': np.random.poisson(1000, n_records),
        'clicks': np.random.poisson(50, n_records),
        'conversions': np.random.poisson(5, n_records),
        'cost': np.random.exponential(100, n_records),
        'revenue': np.random.exponential(500, n_records)
    }
    
    df = pd.DataFrame(data)
    df['ctr'] = df['clicks'] / df['impressions']
    df['conversion_rate'] = df['conversions'] / df['clicks']
    df['roas'] = df['revenue'] / df['cost']
    
    # Cache the data
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save to CSV
        df.to_csv(cache_file, index=False)
        print(f"💾 Data cached to {cache_file}")
        print(f"✅ Generated and cached {len(df)} records")
        print(f"📝 Delete {cache_dir} folder to force regeneration next time")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not cache data to {cache_file}: {e}")
        print("🔄 Continuing with generated data in memory...")
    
    return df

# Tools Definition
def filter_data(query: str, filters: Dict[str, Any], current_data: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Filter dataset based on user criteria"""
    try:
        # Extract filter criteria from query (simplified logic)
        filtered_df = current_data.copy()
        
        # Simple keyword-based filtering
        if 'mobile' in query.lower():
            filtered_df = filtered_df[filtered_df['device'] == 'Mobile']
        if 'google' in query.lower():
            filtered_df = filtered_df[filtered_df['channel'] == 'Google_Ads']
        if 'search' in query.lower():
            filtered_df = filtered_df[filtered_df['campaign'].str.contains('Search')]
        if 'high ctr' in query.lower():
            threshold = filtered_df['ctr'].quantile(0.8)
            filtered_df = filtered_df[filtered_df['ctr'] > threshold]
        
        summary = f"Filtered data: {len(filtered_df)} records (from {len(current_data)} total)"
        print(f"🔍 SUMMARY: {summary}")
        return summary, filtered_df
        
    except Exception as e:
        return f"Error filtering data: {str(e)}", current_data

def analyze_data(query: str, analysis_type: str, current_data: pd.DataFrame) -> str:
    """Perform data analysis and return insights"""
    try:
        if current_data.empty:
            return "No data available for analysis."
        
        insights = []
        
        # Basic statistics
        insights.append("📊 **Data Summary:**")
        insights.append(f"- Total records: {len(current_data):,}")
        insights.append(f"- Date range: {current_data['date'].min().date()} to {current_data['date'].max().date()}")
        
        # Performance metrics
        insights.append("\n💰 **Performance Metrics:**")
        insights.append(f"- Total impressions: {current_data['impressions'].sum():,}")
        insights.append(f"- Total clicks: {current_data['clicks'].sum():,}")
        insights.append(f"- Total conversions: {current_data['conversions'].sum():,}")
        insights.append(f"- Average CTR: {current_data['ctr'].mean():.3%}")
        insights.append(f"- Average Conversion Rate: {current_data['conversion_rate'].mean():.3%}")
        insights.append(f"- Average ROAS: {current_data['roas'].mean():.2f}")
        
        # Top performers
        insights.append("\n🏆 **Top Performers:**")
        top_campaign = current_data.groupby('campaign')['conversions'].sum().idxmax()
        top_channel = current_data.groupby('channel')['revenue'].sum().idxmax()
        insights.append(f"- Best campaign (conversions): {top_campaign}")
        insights.append(f"- Best channel (revenue): {top_channel}")
        
        return "\n".join(insights)
        
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def create_visualization(query: str, chart_type: str, current_data: pd.DataFrame) -> str:
    """Create data visualizations and save them as images"""
    try:
        if current_data.empty:
            return "No data available for visualization."
        
        # Set up matplotlib to work without display (for headless environments)
        plt.ioff()  # Turn off interactive mode
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Determine chart type and filename based on query
        chart_info = ""
        filename = "visualization"
        
        if 'campaign' in query.lower():
            # Campaign performance
            campaign_metrics = current_data.groupby('campaign').agg({
                'impressions': 'sum',
                'clicks': 'sum',
                'conversions': 'sum',
                'revenue': 'sum'
            })
            print("📊 Campaign metrics data:")
            print(campaign_metrics)
            
            # Create subplot for multiple metrics
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            campaign_metrics['impressions'].plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Impressions by Campaign')
            ax1.tick_params(axis='x', rotation=45)
            
            campaign_metrics['clicks'].plot(kind='bar', ax=ax2, color='lightgreen')
            ax2.set_title('Clicks by Campaign')
            ax2.tick_params(axis='x', rotation=45)
            
            campaign_metrics['conversions'].plot(kind='bar', ax=ax3, color='orange')
            ax3.set_title('Conversions by Campaign')
            ax3.tick_params(axis='x', rotation=45)
            
            campaign_metrics['revenue'].plot(kind='bar', ax=ax4, color='lightcoral')
            ax4.set_title('Revenue by Campaign')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.suptitle('Campaign Performance Overview', fontsize=16)
            filename = "campaign_performance"
            chart_info = f"Campaign performance with {len(campaign_metrics)} campaigns"
        
        elif 'time' in query.lower() or 'trend' in query.lower() or 'evolution' in query.lower():
            # Time series
            if 'mobile' in query.lower():
                # Filter for mobile data first
                mobile_data = current_data[current_data['device'] == 'Mobile']
                daily_data = mobile_data.groupby(mobile_data['date'].dt.date).agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'revenue': 'sum'
                })
                title_suffix = " (Mobile Devices)"
                filename = "mobile_trends_over_time"
            else:
                daily_data = current_data.groupby(current_data['date'].dt.date).agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'revenue': 'sum'
                })
                title_suffix = ""
                filename = "trends_over_time"
            
            print("📈 Time series data:")
            print(daily_data.head(10))
            
            # Create subplot for time series
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            daily_data['impressions'].plot(ax=ax1, color='blue', marker='o')
            ax1.set_title(f'Impressions Over Time{title_suffix}')
            ax1.grid(True, alpha=0.3)
            
            daily_data['clicks'].plot(ax=ax2, color='green', marker='s')
            ax2.set_title(f'Clicks Over Time{title_suffix}')
            ax2.grid(True, alpha=0.3)
            
            daily_data['revenue'].plot(ax=ax3, color='red', marker='^')
            ax3.set_title(f'Revenue Over Time{title_suffix}')
            ax3.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            chart_info = f"Time series with {len(daily_data)} data points{title_suffix.lower()}"
        
        else:
            # Default: Channel comparison
            channel_metrics = current_data.groupby('channel')['revenue'].sum()
            print("🥧 Channel metrics data:")
            print(channel_metrics)
            
            # Create pie chart and bar chart side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Pie chart
            channel_metrics.plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Revenue Distribution by Channel')
            ax1.set_ylabel('')  # Remove default ylabel
            
            # Bar chart
            channel_metrics.plot(kind='bar', ax=ax2, color='steelblue')
            ax2.set_title('Revenue by Channel')
            ax2.tick_params(axis='x', rotation=45)
            ax2.set_ylabel('Revenue ($)')
            
            filename = "channel_performance"
            chart_info = f"Channel comparison with {len(channel_metrics)} channels"
        
        plt.tight_layout()
        
        # Save the plot
        cache_dir = "temp_results_agentic_minimal"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(cache_dir, f"{filename}_{timestamp}.png")
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close the figure to free memory
        
        # Get absolute path for user
        abs_filepath = os.path.abspath(filepath)
        
        return f"""📈 **Visualization Created Successfully!**

📊 **Chart Details:**
- Type: {chart_info}
- Records analyzed: {len(current_data):,}
- File saved: `{abs_filepath}`

💡 **How to view:**
- Open the PNG file in any image viewer
- Or open in browser: file:///{abs_filepath.replace(chr(92), '/')}

✅ Chart saved and ready to view!"""
        
    except Exception as e:
        # Make sure to close any open figures in case of error
        plt.close('all')
        return f"Error creating visualization: {str(e)}"

def advertising_knowledge(query: str) -> str:
    """Provide domain knowledge about online advertising"""
    knowledge_base = {
        'ctr': "Click-Through Rate (CTR) is the percentage of people who click on your ad after seeing it. Industry averages: Search ads ~3-5%, Display ads ~0.5-1%, Social media ~1-2%.",
        'conversion': "Conversion rate measures the percentage of users who complete a desired action after clicking your ad. Good conversion rates vary by industry but typically range from 2-5%.",
        'roas': "Return on Ad Spend (ROAS) measures revenue generated for every dollar spent on advertising. A ROAS of 4:1 means $4 revenue for every $1 spent. Target ROAS depends on profit margins.",
        'attribution': "Attribution models help determine which touchpoints deserve credit for conversions. Common models: First-click, Last-click, Linear, Time-decay, Position-based.",
        'audience': "Audience targeting allows you to show ads to specific groups based on demographics, interests, behaviors, or custom data. Lookalike audiences help find users similar to your best customers.",
        'bidding': "Automated bidding strategies use machine learning to optimize bids for your goals: Target CPA (cost per acquisition), Target ROAS, Maximize conversions, etc."
    }
    
    query_lower = query.lower()
    for topic, explanation in knowledge_base.items():
        if topic in query_lower:
            return f"💡 **{topic.upper()}**: {explanation}"
    
    return "🤔 I can help with advertising concepts like CTR, conversion rates, ROAS, attribution, audience targeting, and bidding strategies. Could you be more specific about what you'd like to learn?"

def general_query(query: str, current_data: pd.DataFrame) -> str:
    """Handle general queries about the current dataset"""
    if current_data.empty:
        return "I don't have any data loaded yet. Please load a dataset first or ask me about advertising concepts!"
    
    return f"I have {len(current_data)} records of advertising data available. You can ask me to filter the data, analyze performance, create visualizations, or explain advertising concepts. What would you like to explore?"

# Tool definitions for LLM
TOOL_FILTER_DATA = [{
    "type": "function",
    "function": {
        "name": "filter_data",
        "description": "Filter the dataset based on user criteria (e.g., campaign type, device, channel, performance thresholds)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User's filtering request"},
                "filters": {"type": "object", "description": "Specific filter criteria"}
            },
            "required": ["query"]
        }
    }
}]

TOOL_ANALYZE_DATA = [{
    "type": "function", 
    "function": {
        "name": "analyze_data",
        "description": "Analyze current data and provide insights, statistics, and performance metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Analysis request"},
                "analysis_type": {"type": "string", "description": "Type of analysis (summary, detailed, comparison)"}
            },
            "required": ["query"]
        }
    }
}]

TOOL_CREATE_VISUALIZATION = [{
    "type": "function",
    "function": {
        "name": "create_visualization", 
        "description": "Create charts and visualizations from the current dataset",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Visualization request"},
                "chart_type": {"type": "string", "description": "Type of chart (bar, line, pie, scatter)"}
            },
            "required": ["query"]
        }
    }
}]

TOOL_ADVERTISING_KNOWLEDGE = [{
    "type": "function",
    "function": {
        "name": "advertising_knowledge",
        "description": "Provide domain expertise about online advertising, marketing metrics, and best practices",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question about advertising concepts"}
            },
            "required": ["query"]
        }
    }
}]

TOOL_GENERAL_QUERY = [{
    "type": "function",
    "function": {
        "name": "general_query",
        "description": "Handle general questions about the current data or provide guidance",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "General question or request"}
            },
            "required": ["query"]
        }
    }
}]

# Context Management
@dataclass
class AnalysisContext:
    """Maintains conversation state and data context"""
    current_data: pd.DataFrame = None
    last_filter: str = ""
    analysis_history: List[str] = None
    
    def __post_init__(self):
        if self.analysis_history is None:
            self.analysis_history = []
        if self.current_data is None:
            self.current_data = pd.DataFrame()

# Agent Orchestrator
class DataAnalysisAgent:
    """
    Main orchestrator that routes user queries to appropriate tools
    and maintains conversation context.
    """
    
    def __init__(self):
        self.context = AnalysisContext()
        self.conversation_history = []
        
        # Available tools
        self.tools = {
            'filter_data': filter_data,
            'analyze_data': analyze_data, 
            'create_visualization': create_visualization,
            'advertising_knowledge': advertising_knowledge,
            'general_query': general_query
        }
        
        # Tool definitions for LLM
        self.all_tools = (TOOL_FILTER_DATA + TOOL_ANALYZE_DATA + 
                         TOOL_CREATE_VISUALIZATION + TOOL_ADVERTISING_KNOWLEDGE + 
                         TOOL_GENERAL_QUERY)
        
        # Bind tools to LLM client for function calling
        self.llm_with_tools = llm_client.bind_tools(
            tools=self.all_tools,
            tool_choice="auto"
        )
        
        # Load sample data
        self.context.current_data = generate_sample_data()
        self.llm_call_count = 0  # Track LLM calls manually
        
    def process_user_input(self, user_input: str) -> str:
        """
        Main entry point: processes user input and returns response
        """
        logger.info(f"Processing user input: {user_input}")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Build messages for LLM
            messages = [
                SystemMessage("""You are an intelligent data analysis assistant for advertising and marketing data.
                
                Your role is to help data analysts by:
                - Filtering and analyzing datasets
                - Creating visualizations  
                - Providing domain knowledge about online advertising
                - Answering questions about data insights
                
                Choose the most appropriate tool for each user request. Always be helpful and provide actionable insights.""")
            ]
            
            # Add conversation history (simplified)
            for msg in self.conversation_history[-5:]:  # Last 5 messages
                messages.append(HumanMessage(msg["content"]))
            
            # Get LLM response with tool calling
            completion = self.llm_with_tools.invoke(messages)
            
            response_parts = []
            
            # Handle tool calls
            if hasattr(completion, 'tool_calls') and completion.tool_calls:
                for tool_call in completion.tool_calls:
                    function_name = tool_call['name'] if isinstance(tool_call, dict) else tool_call.name
                    tool_args = tool_call['args'] if isinstance(tool_call, dict) else tool_call.args

                    print(f"🔧 Tool call detected: {function_name} with args: {tool_args}")
                    
                    logger.info(f"Calling tool: {function_name} with args: {tool_args}")
                    
                    # Execute the tool
                    result = self._execute_tool(function_name, tool_args)
                    response_parts.append(result)
            
            # Combine responses
            final_response = "\n\n".join(response_parts) if response_parts else "I'm not sure how to help with that. Could you try rephrasing your question?"
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": final_response})
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return f"I encountered an error: {str(e)}. Please try rephrasing your request."
    
    def _execute_tool(self, function_name: str, tool_args: Dict) -> str:
        """Execute the specified tool with given arguments"""
        
        if function_name not in self.tools:
            return f"Unknown tool: {function_name}"
        
        try:
            if function_name == 'filter_data':
                result, filtered_data = self.tools[function_name](
                    tool_args.get('query', ''),
                    tool_args.get('filters', {}),
                    self.context.current_data
                )
                self.context.current_data = filtered_data
                self.context.last_filter = tool_args.get('query', '')
                return result
                
            elif function_name in ['analyze_data', 'create_visualization']:
                return self.tools[function_name](
                    tool_args.get('query', ''),
                    tool_args.get('analysis_type', 'summary') if function_name == 'analyze_data' else tool_args.get('chart_type', 'auto'),
                    self.context.current_data
                )
                
            elif function_name == 'advertising_knowledge':
                return self.tools[function_name](tool_args.get('query', ''))
                
            elif function_name == 'general_query':
                return self.tools[function_name](
                    tool_args.get('query', ''),
                    self.context.current_data
                )
                
            else:
                return "Tool execution not implemented"
                
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {str(e)}")
            return f"Error executing {function_name}: {str(e)}"
    
    def get_data_info(self) -> str:
        """Get information about current dataset"""
        if self.context.current_data.empty:
            return "No data currently loaded."
        
        return f"""
        📊 **Current Dataset Info:**
        - Records: {len(self.context.current_data):,}
        - Columns: {', '.join(self.context.current_data.columns)}
        - Date range: {self.context.current_data['date'].min().date()} to {self.context.current_data['date'].max().date()}
        """

# Demo Functions
def run_demo():
    """Run a demonstration of the agentic program"""
    print("🤖 Data Analysis Agent Demo")
    print("=" * 50)
    
    agent = DataAnalysisAgent()
    
    # Demo queries
    demo_queries = [
        # "Show me data for mobile devices only",
        # "What's the overall performance of our campaigns?",
        # "Show me data for mobile devices only",
        # "Analyze the current filtered data",
        # "What is CTR and how can I improve it?",
        "Create a visualization of the data: show me an evolution of mobile data over time",
        # "Filter for Google Ads with high conversion rates"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        response = agent.process_user_input(query)
        print(response)
        print("\n" + "="*50)

def interactive_mode():
    """Run the agent in interactive mode"""
    print("🤖 Interactive Data Analysis Agent")
    print("Type 'quit' to exit, 'help' for guidance, 'data' for dataset info")
    print("-" * 60)
    
    agent = DataAnalysisAgent()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("""
                🆘 **Help - What you can ask:**
                
                **Data Operations:**
                - "Filter data for mobile devices"
                - "Show me Google Ads campaigns"
                - "Filter for high CTR campaigns"
                
                **Analysis:**
                - "Analyze the current data"
                - "What are the performance metrics?"
                - "Show me top performing campaigns"
                
                **Visualizations:**
                - "Create a chart of campaign performance"
                - "Visualize trends over time"
                - "Show revenue by channel"
                
                **Advertising Knowledge:**
                - "What is CTR?"
                - "How do I improve conversion rates?"
                - "Explain ROAS"
                """)
                continue
            elif user_input.lower() == 'data':
                print(agent.get_data_info())
                continue
            elif not user_input:
                continue
            
            print(f"\n🤖 Agent: {agent.process_user_input(user_input)}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    print(__doc__)
    
    # Uncomment the mode you want to run:
    
    # Demo mode (automated demonstration)
    run_demo()
    
    # Interactive mode (chat with the agent)
    # interactive_mode()
    
    print("\n✅ Demo completed! Try interactive_mode() to chat with the agent.")
