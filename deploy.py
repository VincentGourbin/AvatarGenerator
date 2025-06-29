#!/usr/bin/env python3
"""
HuggingFace Spaces Deployment Script for Avatar Generator
Deploys the Avatar Generator to HuggingFace Spaces with ZeroGPU support
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import tempfile
import getpass

# Configuration
SPACE_NAME = "VincentGOURBIN/IceBreaker-Avator-Generator"
SPACE_TITLE = "🎭 IceBreaker Avatar Generator"
SPACE_DESCRIPTION = "Generate personalized avatars from Chinese portraits"

def check_dependencies():
    """Check if required tools are installed"""
    print("🔍 Checking dependencies...")
    
    # Check git
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("✅ Git is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install git first.")
        sys.exit(1)
    
    # Check huggingface_hub
    try:
        import huggingface_hub
        print("✅ HuggingFace Hub is available")
    except ImportError:
        print("❌ HuggingFace Hub not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        print("✅ HuggingFace Hub installed")

def get_hf_token():
    """Get HuggingFace token from environment or user input"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if not token:
        print("\n🔑 HuggingFace token required for deployment")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        print("Make sure your token has 'write' permissions")
        token = getpass.getpass("Enter your HuggingFace token: ").strip()
    
    if not token:
        print("❌ No token provided. Deployment cancelled.")
        sys.exit(1)
    
    return token

def create_space_config():
    """Create the Space configuration files"""
    print("📝 Creating Space configuration...")
    
    # Create README.md for the space
    readme_content = f"""---
title: {SPACE_TITLE}
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: mit
hardware: zerogpu
short_description: {SPACE_DESCRIPTION}
models:
- black-forest-labs/FLUX.1-schnell
- google/gemma-3n-E2B-it
tags:
- avatar
- image-generation
- flux
- chinese-portrait
- character-design
- ai-art
---

# 🎭 IceBreaker Avatar Generator

Generate unique avatars based on Chinese portrait descriptions! This app combines FLUX.1-schnell for image generation and Gemma-3n-E2B-it for intelligent prompt optimization.

## Features

🎨 **Dual Generation Modes**:
- **📝 Form Mode**: Direct input with customizable "If I was / I would be" groups
- **💬 Chat Mode**: Simple conversation using randomized categories to discover your Chinese portrait

⚡ **Quality Options**:
- **Normal**: 512x512, 4 steps (fast)
- **High**: 512x512, 8 steps (enhanced quality)

🤖 **AI Models**:
- **FLUX.1-schnell**: High-quality image generation
- **Gemma-3n-E2B-it**: Intelligent prompt optimization for FLUX

🎯 **Smart Chat System**:
- **100+ categories**: Varied questions from animals to philosophical concepts
- **No repetition**: Each category asked only once per conversation
- **Instant responses**: Simple logic without AI overhead for chat flow

## How to Use

### Form Mode
1. Complete at least the first 3 groups (mandatory)
2. Add optional groups 4-5 for richer portraits
3. Select quality and generate your avatar

### Chat Mode
1. Click "🚀 Start Conversation"
2. Follow the AI assistant's guided questions
3. Click "🎨 Get My Avatar" when ready

## Example Chinese Portrait
*If I was an animal, I would be a majestic wolf*
*If I was a color, I would be deep purple*
*If I was an object, I would be an ancient sword*

Try it now and discover your unique avatar! ✨
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    # Create requirements.txt specifically for Spaces
    requirements_content = """gradio[mcp]>=5.34.0
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.24.0
accelerate>=0.24.0
transformers>=4.35.0
sentencepiece>=0.1.97
protobuf>=3.20.0
tokenizers>=0.15.0
timm>=0.9.0
safetensors>=0.3.0
pillow>=9.0.0
numpy>=1.21.0
spaces
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ Space configuration created")

def validate_files():
    """Validate that all required files exist"""
    print("🔍 Validating files...")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "LICENSE"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Check if app.py has ZeroGPU decorators
    with open("app.py", "r") as f:
        app_content = f.read()
        if "@spaces.GPU()" not in app_content:
            print("⚠️ Warning: No @spaces.GPU() decorators found in app.py")
            print("Make sure your GPU-intensive functions have the decorator for ZeroGPU")
        else:
            print("✅ ZeroGPU decorators found")
    
    print("✅ All required files present")

def create_space(token):
    """Create or update the HuggingFace Space"""
    print(f"🚀 Creating/updating Space: {SPACE_NAME}")
    
    from huggingface_hub import HfApi, login
    
    # Login to HuggingFace
    login(token=token, add_to_git_credential=True)
    
    api = HfApi()
    
    try:
        # Try to get space info (check if it exists)
        space_info = api.space_info(repo_id=SPACE_NAME)
        print(f"✅ Space {SPACE_NAME} already exists, updating...")
        update_mode = True
    except Exception:
        print(f"📦 Creating new Space: {SPACE_NAME}")
        update_mode = False
        
        # Create the space
        try:
            api.create_repo(
                repo_id=SPACE_NAME,
                repo_type="space",
                space_sdk="gradio",
                space_hardware="zerogpu",
                private=False
            )
            print("✅ Space created successfully")
        except Exception as e:
            print(f"❌ Failed to create space: {e}")
            sys.exit(1)
    
    return update_mode

def deploy_files(token):
    """Deploy files to the Space"""
    print("📤 Uploading files to Space...")
    
    from huggingface_hub import HfApi
    
    api = HfApi()
    
    # Files to upload
    files_to_upload = [
        "app.py",
        "requirements.txt", 
        "README.md",
        "LICENSE"
    ]
    
    try:
        for file in files_to_upload:
            if os.path.exists(file):
                print(f"  📄 Uploading {file}...")
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=SPACE_NAME,
                    repo_type="space",
                    token=token
                )
            else:
                print(f"  ⚠️ Skipping missing file: {file}")
        
        print("✅ All files uploaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to upload files: {e}")
        sys.exit(1)

def wait_for_space_build():
    """Wait for the space to build"""
    print("⏳ Space is building... This may take a few minutes.")
    print(f"🌐 You can monitor the build at: https://huggingface.co/spaces/{SPACE_NAME}")
    print("📱 The space will be available once the build completes.")

def main():
    """Main deployment function"""
    print("🎭 Avatar Generator - HuggingFace Spaces Deployment")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check dependencies
    check_dependencies()
    
    # Get HuggingFace token
    token = get_hf_token()
    
    # Create space configuration
    create_space_config()
    
    # Validate files
    validate_files()
    
    # Create or update space
    update_mode = create_space(token)
    
    # Deploy files
    deploy_files(token)
    
    # Success message
    print("\n🎉 Deployment completed successfully!")
    print(f"🌐 Space URL: https://huggingface.co/spaces/{SPACE_NAME}")
    
    if not update_mode:
        wait_for_space_build()
    
    print(f"\n📱 Your Avatar Generator is now live at:")
    print(f"   https://huggingface.co/spaces/{SPACE_NAME}")
    print(f"\n🚀 Space deployed with ZeroGPU for optimal performance!")
    print("\n🎭 Happy avatar generating! ✨")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)