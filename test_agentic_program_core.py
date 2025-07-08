"""
Minimalistic Agentic Program for Data Analysis (Core Version)
===========================================================

This program demonstrates how to build an agent that can autonomously decide 
what actions to take based on user input. It's designed for data analysts 
working with advertising/marketing data.

This core version focuses on the agentic pattern without heavy dependencies.

The agent can:
- Filter and analyze datasets (using native Python)
- Answer questions about data insights
- Provide domain knowledge about online advertising
- Route queries to appropriate tools

Architecture:
- Agent Orchestrator: Routes user queries to appropriate tools
- Tools: Modular functions that perform specific tasks
- Context: Maintains conversation state and data context
"""

import logging
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

# Real LLM client using Azure OpenAI (same as backend)
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

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

# Simple data structure for demonstration
class SimpleDataFrame:
    """Simplified DataFrame-like structure using native Python"""
    
    def __init__(self, data=None):
        self.data = data or {}
        self._length = 0
        if data:
            self._length = len(next(iter(data.values())))
    
    def __len__(self):
        return self._length
    
    @property
    def columns(self):
        return list(self.data.keys())
    
    def filter(self, condition_func):
        """Filter data based on condition function"""
        if not self.data:
            return SimpleDataFrame()
        
        indices = []
        for i in range(self._length):
            row = {col: self.data[col][i] for col in self.columns}
            if condition_func(row):
                indices.append(i)
        
        filtered_data = {}
        for col in self.columns:
            filtered_data[col] = [self.data[col][i] for i in indices]
        
        return SimpleDataFrame(filtered_data)
    
    def group_by_sum(self, group_col, sum_col):
        """Simple group by and sum operation"""
        if group_col not in self.data or sum_col not in self.data:
            return {}
        
        groups = {}
        for i in range(self._length):
            key = self.data[group_col][i]
            value = self.data[sum_col][i]
            groups[key] = groups.get(key, 0) + value
        
        return groups
    
    def get_column_stats(self, col):
        """Get basic statistics for a column"""
        if col not in self.data:
            return {}
        
        values = [v for v in self.data[col] if isinstance(v, (int, float))]
        if not values:
            return {}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }

# Sample advertising data generation with caching
def generate_sample_data() -> SimpleDataFrame:
    """
    Generate or load cached sample advertising campaign data.
    Data is cached in temp_results_agentic_minimal folder.
    Delete the folder to force regeneration.
    """
    import os
    import json
    
    # Define cache directory and file path
    cache_dir = "temp_results_agentic_minimal"
    cache_file = os.path.join(cache_dir, "sample_advertising_data.json")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            print(f"📁 Loading cached data from {cache_file}")
            with open(cache_file, 'r') as f:
                data = json.load(f)
            df = SimpleDataFrame(data)
            print(f"✅ Loaded {len(df)} records from cache")
            return df
        except Exception as e:
            print(f"⚠️ Error loading cached data: {e}")
            print("🔄 Generating new data...")
    
    # Generate new data if cache doesn't exist or failed to load
    print(f"🔄 Generating new sample data (cache not found at {cache_file})")
    
    random.seed(42)
    n_records = 1000
    
    campaigns = ['Search_Brand', 'Display_Retargeting', 'Social_Video', 'Search_Generic', 'Display_Prospecting']
    channels = ['Google_Ads', 'Facebook', 'LinkedIn', 'YouTube', 'Bing']
    devices = ['Desktop', 'Mobile', 'Tablet']
    
    data = {
        'date': [(datetime(2024, 1, 1) + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i in range(n_records)],
        'campaign': [random.choice(campaigns) for _ in range(n_records)],
        'channel': [random.choice(channels) for _ in range(n_records)],
        'device': [random.choice(devices) for _ in range(n_records)],
        'impressions': [random.randint(100, 2000) for _ in range(n_records)],
        'clicks': [random.randint(5, 100) for _ in range(n_records)],
        'conversions': [random.randint(0, 10) for _ in range(n_records)],
        'cost': [round(random.uniform(10, 200), 2) for _ in range(n_records)],
        'revenue': [round(random.uniform(50, 1000), 2) for _ in range(n_records)]
    }
    
    # Calculate derived metrics
    data['ctr'] = [round(c/i, 4) if i > 0 else 0 for c, i in zip(data['clicks'], data['impressions'])]
    data['conversion_rate'] = [round(conv/c, 4) if c > 0 else 0 for conv, c in zip(data['conversions'], data['clicks'])]
    data['roas'] = [round(r/cost, 2) if cost > 0 else 0 for r, cost in zip(data['revenue'], data['cost'])]
    
    df = SimpleDataFrame(data)
    
    # Cache the data
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save to JSON
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Data cached to {cache_file}")
        print(f"✅ Generated and cached {len(df)} records")
        print(f"📝 Delete {cache_dir} folder to force regeneration next time")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not cache data to {cache_file}: {e}")
        print("🔄 Continuing with generated data in memory...")
    
    return df

# Tools Definition
def filter_data(query: str, filters: Dict[str, Any], current_data: SimpleDataFrame) -> Tuple[str, SimpleDataFrame]:
    """Filter dataset based on user criteria"""
    try:
        # Extract filter criteria from query (simplified logic)
        def condition(row):
            # Simple keyword-based filtering
            if 'mobile' in query.lower() and row['device'] != 'Mobile':
                return False
            if 'google' in query.lower() and row['channel'] != 'Google_Ads':
                return False
            if 'search' in query.lower() and 'Search' not in row['campaign']:
                return False
            if 'high ctr' in query.lower():
                # Consider CTR > 0.05 as high
                return row['ctr'] > 0.05
            if 'high conversion' in query.lower():
                return row['conversion_rate'] > 0.1
            return True
        
        filtered_df = current_data.filter(condition)
        summary = f"Filtered data: {len(filtered_df)} records (from {len(current_data)} total)"
        return summary, filtered_df
        
    except Exception as e:
        return f"Error filtering data: {str(e)}", current_data

def analyze_data(query: str, analysis_type: str, current_data: SimpleDataFrame) -> str:
    """Perform data analysis and return insights"""
    try:
        if len(current_data) == 0:
            return "No data available for analysis."
        
        insights = []
        
        # Basic statistics
        insights.append("📊 **Data Summary:**")
        insights.append(f"- Total records: {len(current_data):,}")
        
        # Performance metrics
        insights.append("\n💰 **Performance Metrics:**")
        
        # Calculate totals and averages
        imp_stats = current_data.get_column_stats('impressions')
        click_stats = current_data.get_column_stats('clicks')
        conv_stats = current_data.get_column_stats('conversions')
        ctr_stats = current_data.get_column_stats('ctr')
        conv_rate_stats = current_data.get_column_stats('conversion_rate')
        roas_stats = current_data.get_column_stats('roas')
        
        insights.append(f"- Total impressions: {imp_stats.get('sum', 0):,}")
        insights.append(f"- Total clicks: {click_stats.get('sum', 0):,}")
        insights.append(f"- Total conversions: {conv_stats.get('sum', 0):,}")
        insights.append(f"- Average CTR: {ctr_stats.get('avg', 0):.3%}")
        insights.append(f"- Average Conversion Rate: {conv_rate_stats.get('avg', 0):.3%}")
        insights.append(f"- Average ROAS: {roas_stats.get('avg', 0):.2f}")
        
        # Top performers
        insights.append("\n🏆 **Top Performers:**")
        campaign_conversions = current_data.group_by_sum('campaign', 'conversions')
        channel_revenue = current_data.group_by_sum('channel', 'revenue')
        
        if campaign_conversions:
            top_campaign = max(campaign_conversions, key=campaign_conversions.get)
            insights.append(f"- Best campaign (conversions): {top_campaign}")
        
        if channel_revenue:
            top_channel = max(channel_revenue, key=channel_revenue.get)
            insights.append(f"- Best channel (revenue): {top_channel}")
        
        return "\n".join(insights)
        
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

def create_visualization(query: str, chart_type: str, current_data: SimpleDataFrame) -> str:
    """Create data visualizations (text-based for this demo)"""
    try:
        if len(current_data) == 0:
            return "No data available for visualization."
        
        # Simple text-based visualization
        if 'campaign' in query.lower():
            # Campaign performance
            campaign_revenue = current_data.group_by_sum('campaign', 'revenue')
            viz_text = "📊 **Campaign Revenue Visualization:**\n"
            
            if campaign_revenue:
                max_revenue = max(campaign_revenue.values())
                for campaign, revenue in sorted(campaign_revenue.items(), key=lambda x: x[1], reverse=True):
                    bar_length = int((revenue / max_revenue) * 30)
                    bar = "█" * bar_length
                    viz_text += f"{campaign:20} {bar} ${revenue:,.0f}\n"
            
            return viz_text
        
        elif 'channel' in query.lower():
            # Channel comparison
            channel_clicks = current_data.group_by_sum('channel', 'clicks')
            viz_text = "📊 **Channel Clicks Visualization:**\n"
            
            if channel_clicks:
                max_clicks = max(channel_clicks.values())
                for channel, clicks in sorted(channel_clicks.items(), key=lambda x: x[1], reverse=True):
                    bar_length = int((clicks / max_clicks) * 30)
                    bar = "█" * bar_length
                    viz_text += f"{channel:15} {bar} {clicks:,}\n"
            
            return viz_text
        
        else:
            # Default: Device distribution
            device_impressions = current_data.group_by_sum('device', 'impressions')
            viz_text = "📊 **Device Impressions Visualization:**\n"
            
            if device_impressions:
                total = sum(device_impressions.values())
                for device, impressions in sorted(device_impressions.items(), key=lambda x: x[1], reverse=True):
                    percentage = (impressions / total) * 100
                    bar_length = int(percentage / 3)
                    bar = "█" * bar_length
                    viz_text += f"{device:10} {bar} {percentage:.1f}% ({impressions:,})\n"
            
            return viz_text
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

def advertising_knowledge(query: str) -> str:
    """Provide domain knowledge about online advertising"""
    knowledge_base = {
        'ctr': "Click-Through Rate (CTR) is the percentage of people who click on your ad after seeing it. Industry averages: Search ads ~3-5%, Display ads ~0.5-1%, Social media ~1-2%.",
        'conversion': "Conversion rate measures the percentage of users who complete a desired action after clicking your ad. Good conversion rates vary by industry but typically range from 2-5%.",
        'roas': "Return on Ad Spend (ROAS) measures revenue generated for every dollar spent on advertising. A ROAS of 4:1 means $4 revenue for every $1 spent. Target ROAS depends on profit margins.",
        'attribution': "Attribution models help determine which touchpoints deserve credit for conversions. Common models: First-click, Last-click, Linear, Time-decay, Position-based.",
        'audience': "Audience targeting allows you to show ads to specific groups based on demographics, interests, behaviors, or custom data. Lookalike audiences help find users similar to your best customers.",
        'bidding': "Automated bidding strategies use machine learning to optimize bids for your goals: Target CPA (cost per acquisition), Target ROAS, Maximize conversions, etc.",
        'quality score': "Quality Score (Google Ads) measures ad relevance, expected CTR, and landing page experience. Higher scores lead to lower costs and better ad positions.",
        'frequency': "Ad frequency is how often the same person sees your ad. Too high frequency can lead to ad fatigue, while too low may not drive awareness.",
        'impression share': "Impression Share is the percentage of impressions your ads received compared to total available impressions. Low share may indicate budget or bid constraints."
    }
    
    query_lower = query.lower()
    for topic, explanation in knowledge_base.items():
        if topic in query_lower:
            return f"💡 **{topic.upper().replace('_', ' ')}**: {explanation}"
    
    return "🤔 I can help with advertising concepts like CTR, conversion rates, ROAS, attribution, audience targeting, bidding strategies, quality score, frequency, and impression share. Could you be more specific about what you'd like to learn?"

def general_query(query: str, current_data: SimpleDataFrame) -> str:
    """Handle general queries about the current dataset"""
    if len(current_data) == 0:
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
    current_data: SimpleDataFrame = None
    last_filter: str = ""
    analysis_history: List[str] = None
    
    def __post_init__(self):
        if self.analysis_history is None:
            self.analysis_history = []
        if self.current_data is None:
            self.current_data = SimpleDataFrame()

# Agent Orchestrator
class DataAnalysisAgent:
    """
    Main orchestrator that routes user queries to appropriate tools
    and maintains conversation context.
    
    This demonstrates the core agentic pattern:
    1. Receive user input
    2. Analyze intent using LLM
    3. Route to appropriate tool
    4. Execute tool and return results
    5. Maintain context across interactions
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
        logger.info(f"Initialized agent with {len(self.context.current_data)} sample records")
        
    def process_user_input(self, user_input: str) -> str:
        """
        Main entry point: processes user input and returns response
        
        This method demonstrates the core agentic workflow:
        1. Add user input to conversation history
        2. Build context for LLM (system prompt + conversation history)
        3. Let LLM decide which tool to use
        4. Execute the chosen tool(s)
        5. Return formatted response
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
                
                Choose the most appropriate tool for each user request. Always be helpful and provide actionable insights.
                
                Available tools:
                - filter_data: Filter dataset based on criteria
                - analyze_data: Analyze current data and provide insights
                - create_visualization: Create charts and graphs
                - advertising_knowledge: Provide domain expertise
                - general_query: Handle general questions
                """)
            ]
            
            # Add conversation history (last 5 messages for context)
            for msg in self.conversation_history[-5:]:
                messages.append(HumanMessage(msg["content"]))
            
            # Get LLM response with tool calling
            self.llm_call_count += 1
            completion = self.llm_with_tools.invoke(messages)
            
            response_parts = []
            
            # Handle tool calls
            if hasattr(completion, 'tool_calls') and completion.tool_calls:
                for tool_call in completion.tool_calls:
                    function_name = tool_call['name'] if isinstance(tool_call, dict) else tool_call.name
                    tool_args = tool_call['args'] if isinstance(tool_call, dict) else tool_call.args
                    
                    logger.info(f"🔧 Calling tool: {function_name} with args: {tool_args}")
                    
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
        """
        Execute the specified tool with given arguments
        
        This method demonstrates the tool execution pattern:
        - Validate tool exists
        - Call tool with appropriate arguments
        - Handle tool-specific logic (e.g., updating context)
        - Return results
        """
        
        if function_name not in self.tools:
            return f"❌ Unknown tool: {function_name}"
        
        try:
            if function_name == 'filter_data':
                result, filtered_data = self.tools[function_name](
                    tool_args.get('query', ''),
                    tool_args.get('filters', {}),
                    self.context.current_data
                )
                # Update context with filtered data
                self.context.current_data = filtered_data
                self.context.last_filter = tool_args.get('query', '')
                self.context.analysis_history.append(f"Filtered: {tool_args.get('query', '')}")
                return f"✅ {result}"
                
            elif function_name in ['analyze_data', 'create_visualization']:
                result = self.tools[function_name](
                    tool_args.get('query', ''),
                    tool_args.get('analysis_type', 'summary') if function_name == 'analyze_data' else tool_args.get('chart_type', 'auto'),
                    self.context.current_data
                )
                self.context.analysis_history.append(f"Analyzed: {tool_args.get('query', '')}")
                return result
                
            elif function_name == 'advertising_knowledge':
                result = self.tools[function_name](tool_args.get('query', ''))
                return result
                
            elif function_name == 'general_query':
                result = self.tools[function_name](
                    tool_args.get('query', ''),
                    self.context.current_data
                )
                return result
                
            else:
                return f"❌ Tool execution not implemented for {function_name}"
                
        except Exception as e:
            logger.error(f"Error executing tool {function_name}: {str(e)}")
            return f"❌ Error executing {function_name}: {str(e)}"
    
    def get_data_info(self) -> str:
        """Get information about current dataset"""
        if len(self.context.current_data) == 0:
            return "No data currently loaded."
        
        return f"""
📊 **Current Dataset Info:**
- Records: {len(self.context.current_data):,}
- Columns: {', '.join(self.context.current_data.columns)}
- Last filter: {self.context.last_filter or 'None'}
- Analysis history: {len(self.context.analysis_history)} operations
        """
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation"""
        return f"""
💬 **Conversation Summary:**
- Total messages: {len(self.conversation_history)}
- LLM calls: {self.llm_call_count}
- Analysis operations: {len(self.context.analysis_history)}
        """

# Demo Functions
def run_demo():
    """Run a demonstration of the agentic program"""
    print("🤖 Data Analysis Agent Demo")
    print("=" * 50)
    print("This demo shows how an AI agent can autonomously decide what actions to take.")
    print("The agent will route each query to the most appropriate tool.\n")
    
    agent = DataAnalysisAgent()
    
    # Demo queries that showcase different capabilities
    demo_queries = [
        "What's the overall performance of our campaigns?",
        "Show me data for mobile devices only", 
        "Analyze the current filtered data",
        "What is CTR and how can I improve it?",
        "Create a visualization of campaign performance",
        "Filter for Google Ads with high conversion rates",
        "Show me channel performance visualization"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        print("🤖 Agent thinking... (analyzing intent and choosing tool)")
        response = agent.process_user_input(query)
        print(f"🤖 Agent response:\n{response}")
        print("\n" + "="*50)
    
    print(f"\n📈 Demo completed! The agent processed {len(demo_queries)} queries.")
    print(agent.get_conversation_summary())

def interactive_mode():
    """Run the agent in interactive mode"""
    print("🤖 Interactive Data Analysis Agent")
    print("This agent can autonomously decide what to do based on your input!")
    print("\nCommands: 'quit' to exit, 'help' for guidance, 'data' for dataset info, 'summary' for conversation summary")
    print("-" * 80)
    
    agent = DataAnalysisAgent()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("""
🆘 **Help - Example queries (the agent will decide what to do):**

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
- "Visualize channel performance"
- "Show revenue by campaign"

**Advertising Knowledge:**
- "What is CTR?"
- "How do I improve conversion rates?"
- "Explain ROAS and quality score"

**General:**
- "What data do you have?"
- "Help me understand this dataset"
                """)
                continue
            elif user_input.lower() == 'data':
                print(agent.get_data_info())
                continue
            elif user_input.lower() == 'summary':
                print(agent.get_conversation_summary())
                continue
            elif not user_input:
                continue
            
            print(f"\n🤖 Agent: {agent.process_user_input(user_input)}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def explain_architecture():
    """Explain the agentic architecture"""
    print("""
🏗️  **Agentic Architecture Explanation**
=====================================

This program demonstrates a minimalistic agentic system with the following components:

1. **Agent Orchestrator** (DataAnalysisAgent):
   - Receives user input
   - Uses LLM to analyze intent and choose appropriate tool
   - Executes selected tool with extracted parameters
   - Maintains conversation context and data state
   - Returns formatted response

2. **Tools** (Modular Functions):
   - filter_data: Data filtering and subsetting
   - analyze_data: Statistical analysis and insights  
   - create_visualization: Chart and graph generation
   - advertising_knowledge: Domain expertise
   - general_query: Fallback for general questions

3. **Context Management** (AnalysisContext):
   - current_data: Active dataset
   - last_filter: Recent filtering operation
   - analysis_history: Track of performed operations

4. **LLM Integration**:
   - Real Azure OpenAI client (using same credentials as backend)
   - Tool definitions in OpenAI function calling format
   - Conversation history management

**Key Agentic Principles Demonstrated:**

✅ **Autonomy**: Agent decides what action to take
✅ **Tool Use**: Multiple specialized tools available
✅ **Context**: Maintains state across interactions  
✅ **Routing**: Intelligent query-to-tool mapping
✅ **Extensibility**: Easy to add new tools
✅ **Error Handling**: Graceful failure management

**Adaptation for Other Use Cases:**

To adapt this template for other domains:
1. Replace sample data generation with your data
2. Update tools to match your domain needs
3. Modify knowledge base for your domain
4. Replace mock LLM with real LLM client
5. Add domain-specific routing keywords

This pattern scales from simple task routing to complex multi-step workflows!
    """)

if __name__ == "__main__":
    print(__doc__)
    
    # Show architecture explanation
    explain_architecture()
    
    print("\n" + "="*60)
    
    # Uncomment the mode you want to run:
    
    # Demo mode (automated demonstration)
    run_demo()
    
    # Interactive mode (chat with the agent)
    # interactive_mode()
    
    print("\n✅ Demo completed! Try interactive_mode() to chat with the agent.")
    print("💡 This demonstrates how AI agents can autonomously decide what actions to take!")
