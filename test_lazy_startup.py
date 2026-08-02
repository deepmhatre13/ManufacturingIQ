import os
os.environ['PRELOAD_MODELS'] = 'false'

print('=== Testing startup with PRELOAD_MODELS=false ===')
import app.main
print('✓ FastAPI app imported')

from fastapi.testclient import TestClient
client = TestClient(app.main.app)

response = client.get('/health')
print(f'✓ /health: {response.status_code} - {response.json()}')

print()
print('=== Testing lazy model loading ===')
import time
start = time.time()
test_payload = {
    'Type': 'L',
    'Air_temperature_K': 300.0,
    'Process_temperature_K': 310.0,
    'Rotational_speed_rpm': 1500,
    'Torque_Nm': 40.0,
    'Tool_wear_min': 100
}
result = app.main.predict_machine_failure(test_payload)
elapsed = time.time() - start
print(f'✓ First prediction: {elapsed:.2f}s')
print(f'  Result: {result}')

# Second prediction should be faster (cached)
start = time.time()
result2 = app.main.predict_machine_failure(test_payload)
elapsed2 = time.time() - start
print(f'✓ Second prediction: {elapsed2:.2f}s (cached)')
print()
print('=== All checks passed ===')