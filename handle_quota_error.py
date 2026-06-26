"""
Quota Error Handler - Help users understand and fix quota errors
"""
import json
import sys
import io

# Set stdout to UTF-8 encoding (Windows compatibility)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_quota_errors(results_file: str = "evaluation_results.json"):
    """
    Analyze evaluation results and provide guidance on quota errors
    
    Args:
        results_file: Path to evaluation results JSON file
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    
    # Count errors
    quota_errors = 0
    rate_limit_errors = 0
    other_errors = 0
    
    detailed_results = data.get('detailed_results', [])
    
    for result in detailed_results:
        prediction = result.get('prediction', {})
        if isinstance(prediction, dict):
            error = prediction.get('ERROR', '')
            if error == 'QUOTA_EXCEEDED' or 'quota' in str(error).lower():
                quota_errors += 1
            elif error == 'RATE_LIMIT' or 'rate_limit' in str(error).lower():
                rate_limit_errors += 1
            elif 'ERROR' in str(error):
                other_errors += 1
    
    # Print analysis
    print("="*70)
    print("📊 Error Analysis")
    print("="*70)
    print(f"\nQuota Errors: {quota_errors}")
    print(f"Rate Limit Errors: {rate_limit_errors}")
    print(f"Other Errors: {other_errors}")
    
    if quota_errors > 0:
        print("\n" + "="*70)
        print("⚠️  QUOTA EXCEEDED - Action Required")
        print("="*70)
        print("\nYour API quota has been exceeded. Here's what to do:")
        print("\n1. Check Your Billing:")
        model_name = data.get('model', '')
        if 'openai' in model_name.lower() or 'gpt' in model_name.lower():
            print("   → https://platform.openai.com/account/billing")
            print("   → Add payment method or increase quota")
        elif 'claude' in model_name.lower():
            print("   → https://console.anthropic.com/settings/billing")
            print("   → Check your usage limits")
        
        print("\n2. Solutions:")
        print("   • Add payment method to your account")
        print("   • Upgrade your plan")
        print("   • Wait for quota to reset (usually monthly)")
        print("   • Use a different API key with available quota")
        
        print("\n3. Continue Evaluation:")
        print("   • After fixing quota, you can resume from where you left off")
        print("   • Use --subset to test specific items first")
        
    if rate_limit_errors > 0:
        print("\n" + "="*70)
        print("⚠️  RATE LIMIT EXCEEDED")
        print("="*70)
        print("\nYou're making requests too quickly. Solutions:")
        print("   • Increase sleep_time in config.yaml (e.g., 2.0 seconds)")
        print("   • Use --subset to evaluate fewer items at once")
        print("   • Wait a few minutes and try again")
    
    print("\n" + "="*70)


def get_successful_results(results_file: str = "evaluation_results.json"):
    """
    Extract only successful results (excluding errors)
    
    Args:
        results_file: Path to evaluation results JSON file
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    
    detailed_results = data.get('detailed_results', [])
    successful_results = []
    
    for result in detailed_results:
        prediction = result.get('prediction', {})
        if isinstance(prediction, dict):
            error = prediction.get('ERROR', '')
            if not error or error not in ['QUOTA_EXCEEDED', 'RATE_LIMIT']:
                successful_results.append(result)
    
    print(f"✅ Found {len(successful_results)} successful results out of {len(detailed_results)} total")
    return successful_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = "evaluation_results.json"
    
    analyze_quota_errors(results_file)

