import requests
import json
try:
    r = requests.post('http://127.0.0.1:5000/api/ask-budget', 
                     json={'question': 'What agriculture policies are in the budget?', 'language': 'en'},
                     timeout=40)
    print(f'Status: {r.status_code}')
    data = r.json()
    print(f'OK: {data.get("ok")}')
    print(f'Confidence: {data.get("confidence", {})}')
    answer = data.get('answer', '')
    print(f'\n=== ANSWER (first 350 chars) ===')
    print(answer[:350])
    print(f'\n=== Full answer length: {len(answer)} chars ===')
except Exception as e:
    print(f'Error: {type(e).__name__}: {e}')
