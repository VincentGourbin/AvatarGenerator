#!/usr/bin/env python3
"""
Test script for Avatar Generator MCP functionality
"""

import subprocess
import sys
import time
import requests
import json

def test_mcp_server():
    """Test the MCP server functionality"""
    print("🧪 Testing Avatar Generator MCP Server")
    print("=" * 50)
    
    # Start the MCP server
    print("🚀 Starting MCP server...")
    try:
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(10)
        
        # Test if server is running
        try:
            response = requests.get("http://localhost:7860", timeout=5)
            if response.status_code == 200:
                print("✅ MCP server is running!")
                print(f"🌐 Server URL: http://localhost:7860")
                print(f"📡 MCP endpoint: http://localhost:7860/gradio_api/mcp/sse")
                
                # Test API endpoints
                print("\n🔍 Testing API endpoints...")
                
                # Test suggestions endpoint
                try:
                    api_response = requests.post(
                        "http://localhost:7860/api/predict",
                        json={
                            "data": ["fantasy"],
                            "fn_index": 2  # Suggestions interface index
                        },
                        timeout=10
                    )
                    if api_response.status_code == 200:
                        print("✅ Suggestions API working")
                    else:
                        print(f"⚠️ Suggestions API returned: {api_response.status_code}")
                except Exception as e:
                    print(f"❌ Suggestions API error: {e}")
                
            else:
                print(f"❌ Server returned status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Could not connect to server: {e}")
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    finally:
        # Clean up
        if 'process' in locals():
            process.terminate()
            process.wait()
            print("🛑 Server stopped")
    
    return True

def show_mcp_integration_guide():
    """Show integration guide for Claude Desktop"""
    print("\n📋 MCP Integration Guide")
    print("=" * 50)
    
    print("1. 📁 Claude Desktop Configuration:")
    print("   Copy claude_desktop_config.json content to your Claude Desktop config:")
    print("   ~/Library/Application Support/Claude/claude_desktop_config.json")
    
    print("\n2. 🚀 Start MCP Server:")
    print("   python app.py")
    
    print("\n3. 🔧 Available MCP Tools:")
    print("   • generate_avatar: Create avatar from Chinese portrait elements")
    print("   • generate_avatar_from_chat: Generate from conversation history")
    
    print("\n4. 💬 Example Usage in Claude:")
    print("   'Use the avatar generator to create a portrait with a wolf, purple color, and ancient sword'")
    
    print("\n5. 📡 Direct MCP Endpoint:")
    print("   http://localhost:7860/gradio_api/mcp/sse")

if __name__ == "__main__":
    show_mcp_integration_guide()
    
    if "--test" in sys.argv:
        test_mcp_server()
    else:
        print("\n🧪 To run server test: python test_mcp.py --test")