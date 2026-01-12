"""
Quick Test Script - Verify System is Working
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("RFQ Intelligence System - Quick Test")
print("=" * 60)

# Test 1: Health Check
print("\n1. Testing Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        print("   ✓ API is healthy")
        result = response.json()
        print(f"   Ollama Status: {result.get('ollama', 'unknown')}")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")
except Exception as e:
    print(f"   ✗ Cannot connect to API: {e}")
    print("\n   Make sure the server is running:")
    print("   Run: python main.py")
    exit(1)

# Test 2: Simple Text RFQ
print("\n2. Testing Simple Text RFQ...")
try:
    response = requests.post(
        f"{BASE_URL}/api/v1/rfq/text",
        json={
            "text": "Need 10 electric motors, 5HP, 3-phase, 415V",
            "context": "Test request"
        },
        timeout=60
    )
    
    if response.status_code == 200:
        result = response.json()
        print("   ✓ RFQ processed successfully")
        print(f"   Confidence Score: {result['confidence_score']}")
        print(f"   Accuracy: {result['accuracy']}")
        print(f"   RFQ Type: {result['rfq_archetype']}")
        print(f"   Products Found: {len(result['products'])}")
        
        if result['products']:
            product = result['products'][0]
            print(f"\n   First Product:")
            print(f"   - Name: {product['product_name']}")
            print(f"   - Quantity: {product['qty']} {product['unit']}")
            print(f"   - SKU: {product['Sku']}")
            print(f"   - BOM: {product['Bom']}")
            print(f"   - Estimated Cost: {product['Estimated_cost']}")
        
        print(f"\n   AI Suggestion: {result['Ai_generated']['ai_suggestion']}")
        
    else:
        print(f"   ✗ RFQ processing failed: {response.status_code}")
        print(f"   Error: {response.text}")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nNext Steps:")
print("1. Check full API docs: http://localhost:8000/docs")
print("2. Run comprehensive tests: python test_examples.py")
print("3. Try different input formats (PDF, Excel, etc.)")
