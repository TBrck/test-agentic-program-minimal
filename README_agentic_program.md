# Minimalistic Agentic Program for Data Analysis

This repository contains a demonstration of how to build an agentic AI system that can autonomously decide what actions to take based on user input. It's inspired by the conversational-ads backend orchestrator pattern and designed for data analysts working with advertising/marketing data.

## Overview

The program demonstrates key agentic principles:
- **Autonomy**: Agent decides what action to take based on user intent
- **Tool Use**: Multiple specialized tools available for different tasks
- **Context**: Maintains state across interactions
- **Routing**: Intelligent query-to-tool mapping using LLM
- **Extensibility**: Easy to add new tools and capabilities

## Architecture

```
User Input → Agent Orchestrator → LLM Intent Analysis → Tool Selection → Tool Execution → Response
                     ↓
               Context Management
                (Data State, History)
```

### Components

1. **Agent Orchestrator** (`DataAnalysisAgent`):
   - Receives user input
   - Uses LLM to analyze intent and choose appropriate tool
   - Executes selected tool with extracted parameters
   - Maintains conversation context and data state
   - Returns formatted response

2. **Tools** (Modular Functions):
   - `filter_data`: Data filtering and subsetting
   - `analyze_data`: Statistical analysis and insights
   - `create_visualization`: Chart and graph generation
   - `advertising_knowledge`: Domain expertise
   - `general_query`: Fallback for general questions

3. **Context Management** (`AnalysisContext`):
   - `current_data`: Active dataset
   - `last_filter`: Recent filtering operation
   - `analysis_history`: Track of performed operations

4. **LLM Integration**:
   - **Real Azure OpenAI client** (using same credentials as backend)
   - Tool definitions in OpenAI function calling format
   - Conversation history management
   - Actual LLM-powered intent analysis and tool routing

## Files

### Core Version (Recommended for Learning)
- **`test_agentic_program_core.py`**: Core implementation using **real Azure OpenAI LLM**
  - Uses same Azure OpenAI credentials as your backend
  - Simplified data structures for easier understanding
  - Perfect for understanding the agentic pattern
  - Includes architectural explanation and demo
  - **Actually calls the LLM** for tool routing decisions

### Full Version (Production Ready)
- **`test_agentic_program_minimal.py`**: Full implementation with **real Azure OpenAI LLM** + pandas, matplotlib, etc.
- **`requirements_minimal_agent.txt`**: Dependencies for the full version

## Real LLM Integration

Both versions now use **actual Azure OpenAI LLM** with the same credentials as your backend:

```python
# Azure OpenAI configuration (same as backend)
llm_client = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-10-21",
    azure_endpoint="https://cail-ad-chatbot-instance-azure-openai.openai.azure.com/",
    openai_api_key="your-key"
)

# Bind tools for function calling
llm_with_tools = llm_client.bind_tools(tools=all_tools, tool_choice="auto")
```

When you run the program, you'll see actual HTTP requests to Azure OpenAI:
```
INFO:httpx:HTTP Request: POST https://cail-ad-chatbot-instance-azure-openai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-21 "HTTP/1.1 200 OK"
```

The LLM **autonomously decides** which tool to call based on:
- User intent analysis
- Available tool descriptions
- Conversation context
- Function calling capabilities

## Usage

### Running the Core Version (Recommended)

```bash
cd backend
python test_agentic_program_core.py
```

This will:
1. Show an architectural explanation
2. Run an automated demo with 7 example queries
3. Display how the agent routes each query to appropriate tools

### Interactive Mode

To chat with the agent interactively, modify the end of the core file:

```python
if __name__ == "__main__":
    # Comment out the demo and uncomment interactive mode
    # run_demo()
    interactive_mode()
```

### Example Queries

The agent can handle various types of requests:

**Data Operations:**
```
"Filter data for mobile devices"
"Show me Google Ads campaigns"
"Filter for high CTR campaigns"
```

**Analysis:**
```
"Analyze the current data"
"What are the performance metrics?"
"Show me top performing campaigns"
```

**Visualizations:**
```
"Create a chart of campaign performance"
"Visualize channel performance"
"Show revenue by campaign"
```

**Advertising Knowledge:**
```
"What is CTR?"
"How do I improve conversion rates?"
"Explain ROAS and quality score"
```

## Key Features Demonstrated

### 1. Autonomous Decision Making
The agent uses keyword analysis (or LLM in production) to determine the most appropriate tool for each query.

### 2. Tool Execution
Tools are modular functions that can be easily extended or modified:

```python
def new_tool(query: str, parameters: Dict) -> str:
    """Add your custom tool here"""
    # Tool logic
    return result
```

### 3. Context Preservation
The agent maintains data state and conversation history across interactions:

```python
@dataclass
class AnalysisContext:
    current_data: SimpleDataFrame = None
    last_filter: str = ""
    analysis_history: List[str] = None
```

### 4. Error Handling
Graceful error handling ensures the agent continues functioning even when tools fail.

## Adapting to Other Use Cases

To adapt this template for other domains:

1. **Replace Data Generation**: Update `generate_sample_data()` with your data source
2. **Add Domain Tools**: Create new tools specific to your use case
3. **Update Knowledge Base**: Modify `advertising_knowledge()` for your domain
4. **Enhance Routing**: Add domain-specific keywords for tool selection
5. **Integrate Real LLM**: Replace `MockLLMClient` with actual LLM client

### Example: Finance Data Analysis

```python
def financial_knowledge(query: str) -> str:
    knowledge_base = {
        'roi': "Return on Investment measures profitability...",
        'volatility': "Volatility measures price fluctuation...",
        # Add more financial concepts
    }
    # Implementation

def risk_analysis(query: str, data: SimpleDataFrame) -> str:
    # Financial risk analysis tool
    pass
```

## Production Deployment

For production use:

1. **Replace Mock LLM**: Use actual OpenAI, Azure OpenAI, or other LLM service
2. **Add Authentication**: Implement user authentication and session management
3. **Database Integration**: Connect to your actual data sources
4. **Error Monitoring**: Add comprehensive logging and monitoring
5. **API Interface**: Wrap in FastAPI or similar framework
6. **Security**: Add input validation and sanitization

### Example LLM Integration

```python
from langchain_openai import ChatOpenAI

client = ChatOpenAI(
    model="gpt-4",
    api_key="your-api-key"
)

llm_with_tools = client.bind_tools(
    tools=all_tools,
    tool_choice="auto"
)
```

## Benefits of This Pattern

1. **Scalability**: Easy to add new tools and capabilities
2. **Maintainability**: Clean separation of concerns
3. **Flexibility**: Can handle diverse query types
4. **User-Friendly**: Natural language interface
5. **Extensible**: Framework can adapt to many domains

## Learning Path

1. **Start with Core**: Run `test_agentic_program_core.py` to understand the pattern
2. **Explore Architecture**: Read the architectural explanation in the output
3. **Try Interactive Mode**: Chat with the agent to see routing in action
4. **Extend Tools**: Add your own tools and test them
5. **Integrate Real LLM**: Replace mock client with actual LLM service
6. **Build Your Domain**: Adapt for your specific use case

This template provides a solid foundation for building agentic systems that can autonomously decide what actions to take based on user input, making it valuable for data analysts, researchers, and developers working on AI-powered applications.

## Inspiration

This implementation is inspired by the LLM orchestrator pattern used in the conversational-ads backend, demonstrating how production agentic systems route user queries to appropriate tools while maintaining context and providing intelligent responses.
