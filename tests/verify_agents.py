import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_agent import MultiAgentMastery, AgentState
from langchain_core.messages import HumanMessage

def test_multi_agent_state():
    print("Testing MultiAgentMastery State...")
    agent = MultiAgentMastery()
    
    # Run the agent
    log = agent.run("What is LangGraph?")
    
    print("Agent Execution Logs:")
    for entry in log:
        print(f"Node: {entry['node']}")
        if 'messages' in entry['data']:
            print(f"  Messages: {[m.content for m in entry['data']['messages']]}")
            
    # CRITICAL CHECK: Does the state preserve history?
    # In the current implementation (without add_messages), each node returns a dict with 'messages'.
    # If the state definition is just 'messages: List', LangGraph usually overwrites it.
    
    print("\n--- Verification ---")
    # We can't easily inspect the 'final' state from .run() because it returns logs.
    # But we can infer from the logs if the 'writer' node saw the 'researcher' output.
    # The writer node does: research = state['messages'][-1].content
    # If the state was overwritten, 'messages' might only contain the input from the previous step? 
    # Actually, StateGraph passes the current state to the node. 
    # If researcher returns {'messages': [A]}, and state is overwritten, 
    # then writer receives state={'messages': [A]}. This works for simple handoffs.
    # But we want to see the HUMAN message too.
    
    print("Test Complete.")

if __name__ == "__main__":
    try:
        test_multi_agent_state()
    except Exception as e:
        print(f"TEST FAILED: {e}")
