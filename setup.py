import subprocess
import sys
import urllib.request
import urllib.error

def check_ollama_running():
    print("🔍 Checking if Ollama is running in the background...")
    try:
        # Pings the default Ollama local server port
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        print("✅ Ollama is active and responding!")
        return True
    except urllib.error.URLError:
        print("\n❌ ERROR: Ollama is not running.")
        print("Please open the Ollama application on your computer.")
        print("Wait for it to start (look for the icon in your system tray/menu bar), then run this script again.")
        return False

def check_and_pull_model():
    model_name = "llama3.1"
    print(f"\n🔍 Checking if the '{model_name}' brain is downloaded...")
    
    try:
        # Asks Ollama for the list of currently installed models
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        
        if model_name in result.stdout:
            print(f"✅ Model '{model_name}' is already installed! You are good to go.")
        else:
            print(f"⏳ Model '{model_name}' is missing.")
            print(f"⬇️  Downloading now (this is ~4.7GB and may take a few minutes)...")
            
            # Runs the pull command and hands control of the terminal to Ollama 
            # so the user can see the actual download progress bar
            subprocess.run(['ollama', 'pull', model_name], check=True)
            
            print(f"\n🎉 Download complete! '{model_name}' is fully locked and loaded.")
            
    except FileNotFoundError:
        print("\n❌ ERROR: The 'ollama' command was not found.")
        print("Please ensure you installed Ollama from https://ollama.com/")
        sys.exit(1)

def main():
    print("="*55)
    print("🚀 Project Initializer: AI Librarian Environment Setup")
    print("="*55 + "\n")
    
    if not check_ollama_running():
        sys.exit(1)
        
    check_and_pull_model()
    
    print("\n" + "="*55)
    print("✅ System completely configured.")
    print("You can now safely run: python app.py")
    print("="*55)

if __name__ == "__main__":
    main()