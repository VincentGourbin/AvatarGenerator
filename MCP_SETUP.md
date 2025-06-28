# 🔌 MCP Integration Guide

This guide explains how to integrate the Avatar Generator with Claude Desktop using the Model Context Protocol (MCP).

## What is MCP?

The Model Context Protocol allows AI applications like Claude Desktop to connect to external tools and services. With MCP, Claude can directly use the Avatar Generator to create personalized avatars during conversations.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install "gradio[mcp]>=4.40.0"
```

### 2. Start MCP Server
```bash
python app.py
```
*Note: MCP is enabled by default*

### 3. Configure Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "avatar-generator": {
      "command": "python",
      "args": ["/full/path/to/AvatarGenerator/app.py"],
      "env": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1"
      }
    }
  }
}
```

**Important**: Replace `/full/path/to/AvatarGenerator/` with the actual path to your project directory.

### 4. Restart Claude Desktop

After adding the configuration, restart Claude Desktop to load the MCP server.

## 🛠️ Available Tools

The MCP server exposes the main application functions as tools:

### 1. `mcp_create_chinese_portrait`
Create avatars from individual Chinese portrait elements.

**Parameters**:
- `animal`: The animal you would be (e.g., "a majestic wolf")
- `color`: The color you would be (e.g., "deep purple")
- `object_type`: The object you would be (e.g., "an ancient sword")
- `feeling`: Optional feeling (e.g., "fierce determination")
- `element`: Optional element (e.g., "lightning")
- `quality`: "normal" or "high" quality

### 2. `mcp_generate_avatar_from_text`
Generate avatars from complete Chinese portrait descriptions.

**Parameters**:
- `portrait_description`: Full text description
- `quality`: "normal" or "high" quality

### 3. `mcp_get_portrait_suggestions`
Get themed suggestions for creating Chinese portraits.

**Parameters**:
- `theme`: "fantasy", "nature", "modern", or "mystical"

## 💬 Usage Examples

Once configured, you can ask Claude to use the avatar generator:

```
"Create an avatar where I'm a wolf, with purple color, and an ancient sword"

"Generate a Chinese portrait for me based on fantasy themes"

"Give me suggestions for a mystical Chinese portrait and then create one"

"I want to be a dragon, gold color, and a magical crystal - create my avatar"
```

## 🔧 Troubleshooting

### Server Won't Start
- Check that Python and all dependencies are installed
- Verify the path in the Claude Desktop config is correct
- Make sure no other service is using port 7860

### Claude Can't Find Tools
- Restart Claude Desktop after configuration changes
- Check the Claude Desktop logs for connection errors
- Verify the MCP server is running: `curl http://localhost:7860`

### Generation Errors
- Ensure you have sufficient disk space for model downloads
- Check GPU memory if using CUDA
- Try using CPU mode if GPU issues persist

## 🌐 Direct API Access

The MCP server also provides direct API access:

- **Web Interface**: `http://localhost:7860`
- **MCP Endpoint**: `http://localhost:7860/gradio_api/mcp/sse`
- **API Documentation**: `http://localhost:7860/docs`

## 🔄 Testing

Use the provided test script to verify your setup:

```bash
# Show setup guide
python test_mcp.py

# Run full server test
python test_mcp.py --test
```

## 📋 Configuration Template

Copy this template to your Claude Desktop config:

```json
{
  "mcpServers": {
    "avatar-generator": {
      "command": "python",
      "args": ["__PATH_TO_PROJECT__/app.py"],
      "env": {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "GRADIO_SERVER_PORT": "7860"
      }
    }
  }
}
```

Remember to replace `__PATH_TO_PROJECT__` with your actual project path!

## 🎯 Next Steps

1. **Test the Integration**: Try the example prompts above
2. **Experiment**: Create your own Chinese portrait combinations
3. **Share**: Show others your generated avatars
4. **Contribute**: Report issues or suggest improvements

Happy avatar generating! 🎭✨