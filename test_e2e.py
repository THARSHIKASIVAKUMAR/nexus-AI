import requests

def test_backend():
    url = "http://localhost:8001/analyze"
    files = {'file': ('test.txt', 'This is a test file for NexusAI.')}
    data = {'prompt': 'Analyze this test file.'}
    
    try:
        print(f"Testing {url}...")
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        print("Success! Response received:")
        print(response.json())
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_backend()
