"""
API Connection Test Script
Test if your API keys are configured correctly
"""
import sys
import io
import os
from dotenv import load_dotenv

# Set stdout to UTF-8 encoding (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

def test_openai():
    """Test OpenAI API connection"""
    print("\n" + "="*60)
    print("Testing OpenAI API...")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("   Please set it in .env file or environment")
        return False
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        print("✓ OpenAI library imported successfully")
        print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
        
        # Test API call
        print("\nMaking test API call...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✓ API call successful!")
        print(f"✓ Response: {result}")
        print(f"✓ Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except ImportError:
        print("❌ OpenAI library not installed")
        print("   Run: pip install openai")
        return False
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return False


def test_claude():
    """Test Claude API connection"""
    print("\n" + "="*60)
    print("Testing Claude API...")
    print("="*60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("   Please set it in .env file or environment")
        return False
    
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        
        print("✓ Anthropic library imported successfully")
        print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
        
        # Test API call
        print("\nMaking test API call...")
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'Hello' in one word."}]
        )
        
        result = response.content[0].text.strip()
        print(f"✓ API call successful!")
        print(f"✓ Response: {result}")
        print(f"✓ Tokens used: {response.usage.input_tokens + response.usage.output_tokens}")
        
        return True
        
    except ImportError:
        print("❌ Anthropic library not installed")
        print("   Run: pip install anthropic")
        return False
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return False


def main():
    """Main test function"""
    print("🔌 API Connection Test")
    print("="*60)
    print("\nThis script will test your API configurations.")
    print("Make sure you have:")
    print("  1. Created a .env file with your API keys")
    print("  2. Installed required libraries (openai, anthropic)")
    print("\n" + "="*60)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("\n⚠️  Warning: .env file not found")
        print("   Create a .env file with your API keys")
        print("   See .env.example for reference")
    
    results = {
        "OpenAI": False,
        "Claude": False
    }
    
    # Test OpenAI
    try:
        results["OpenAI"] = test_openai()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return
    
    # Test Claude
    try:
        results["Claude"] = test_claude()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    for api, success in results.items():
        status = "✅ Working" if success else "❌ Not configured"
        print(f"  {api}: {status}")
    
    print("\n" + "="*60)
    if any(results.values()):
        print("✅ At least one API is configured correctly!")
        print("\nYou can now run evaluation with:")
        if results["OpenAI"]:
            print("  python main.py --adapter openai --model gpt-3.5-turbo")
        if results["Claude"]:
            print("  python main.py --adapter claude --model claude-3-sonnet-20240229")
    else:
        print("❌ No APIs are configured correctly")
        print("\nPlease:")
        print("  1. Create a .env file (see .env.example)")
        print("  2. Add your API keys")
        print("  3. Install required libraries: pip install openai anthropic")
    print("="*60)


if __name__ == "__main__":
    main()

