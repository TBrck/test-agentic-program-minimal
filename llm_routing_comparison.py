"""
Quick Demo: Real LLM vs Mock Comparison
======================================

This script demonstrates the difference between using a real LLM and a mock client
for agentic decision making.
"""

def mock_routing_demo():
    """Show how mock routing would work (keyword-based)"""
    print("🤖 MOCK ROUTING (keyword-based):")
    print("-" * 40)
    
    test_queries = [
        "Can you filter the data for me?",
        "I want to see a visualization", 
        "Tell me about CTR",
        "What insights can you provide?",
        "Show me some charts"
    ]
    
    for query in test_queries:
        # Simple keyword matching logic
        if any(word in query.lower() for word in ['filter', 'select', 'where']):
            tool = "filter_data"
        elif any(word in query.lower() for word in ['visualize', 'chart', 'graph']):
            tool = "create_visualization"
        elif any(word in query.lower() for word in ['ctr', 'conversion', 'advertising']):
            tool = "advertising_knowledge"
        elif any(word in query.lower() for word in ['insights', 'analyze']):
            tool = "analyze_data"
        else:
            tool = "general_query"
            
        print(f"Query: '{query}' → {tool}")

def real_llm_demo():
    """Show how real LLM routing works"""
    print("\n🧠 REAL LLM ROUTING (context-aware):")
    print("-" * 40)
    print("With real LLM, the agent:")
    print("✅ Understands context and intent")
    print("✅ Can handle complex, nuanced queries")
    print("✅ Makes intelligent tool selection decisions")
    print("✅ Considers conversation history")
    print("✅ Handles ambiguous requests gracefully")
    print("\nExample from actual run:")
    print("Query: 'What's the overall performance of our campaigns?'")
    print("LLM Decision: analyze_data (with specific parameters)")
    print("Tool Args: {'query': 'overall performance of all campaigns', 'analysis_type': 'summary'}")

if __name__ == "__main__":
    print("🔄 AGENTIC ROUTING COMPARISON")
    print("=" * 50)
    
    mock_routing_demo()
    real_llm_demo()
    
    print("\n" + "=" * 50)
    print("💡 The real LLM version provides much more intelligent")
    print("   and context-aware tool routing decisions!")
