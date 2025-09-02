# Category Boosted Recommendations Test Suite

## Overview
Comprehensive test suite for the `recommend_items_category_boosted` function that verifies:
- ✅ **Policy Compliance**: 50% category allocation as documented
- ✅ **Edge Case Handling**: No history, empty categories, invalid IDs
- ✅ **Performance**: Sub-second response times
- ✅ **Output Validation**: Correct format and data types

## Quick Start

### Prerequisites
- Trained recommendation engine models in `src/artifacts/`
- Dataset files: `datasets/users.csv`, `datasets/items.csv`, `datasets/interactions.csv`

### Run Tests
```bash
# Navigate to project directory
cd /home/user/Desktop/RecSys-HP

# Run comprehensive test suite
python test_category_boosted_recommendations.py
```

## Test Coverage

### 🧪 Test 1: Normal Operation
- Tests users with diverse interaction histories
- Verifies category alignment (~50% policy target)
- Validates recommendation quality and scores

### 🧪 Test 2: Edge Cases
- **No History**: Fallback to collaborative filtering
- **Empty History**: Graceful handling of empty lists  
- **Invalid IDs**: Non-existent item IDs in history

### 🧪 Test 3: Policy Compliance
- **50% Category Allocation**: Strict policy verification
- **Multiple User Types**: Electronics, books, mixed categories
- **Tolerance**: ±10% variance allowed (40-60% range)

### 🧪 Test 4: Performance & Validation
- **Speed Test**: 10 users, target <1 second average
- **Output Format**: Correct tuple structure (item_id, score, info)
- **Data Types**: Integer IDs, float scores, dict metadata
- **Required Fields**: product_id, category_code, brand, price

## Expected Output

```
🎯 CATEGORY BOOSTED RECOMMENDATIONS - COMPREHENSIVE TEST SUITE
================================================================================

🧪 TEST 1: Normal Operation with Interaction History
============================================================
👤 Test user: 32-year-old male, $75,000 income
📚 Interaction history: [1000978, 1001588, 1001618, 1002456, 1003789, 1004123]
🏷️  Historical categories: ['books.fiction', 'electronics.smartphone', ...]

⏱️  Execution time: 0.234 seconds
📊 Generated 10 recommendations

📈 POLICY COMPLIANCE ANALYSIS:
   Category Alignment: 52.3%
   Expected: ~50% (policy target)
   Matching items: 5/10

✅ Test Result: PASS

[Additional tests...]

📊 COMPREHENSIVE TEST SUMMARY
================================================================================
✅ PASS     Normal Operation
✅ PASS     Edge Cases  
✅ PASS     Policy Compliance
✅ PASS     Performance & Validation

🎯 OVERALL RESULT: 4/4 tests passed
🎉 ALL TESTS PASSED! The recommend_items_category_boosted function is working correctly.
```

## Troubleshooting

### Common Issues

1. **Missing Artifacts**: Ensure models are trained first
   ```bash
   python run_training_pipeline.py
   ```

2. **Dataset Issues**: Verify CSV files exist and have correct format
   ```bash
   ls -la datasets/*.csv
   ```

3. **Import Errors**: Ensure PYTHONPATH includes src directory
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

### Performance Issues
- If tests are slow (>1 second), check FAISS index loading
- GPU acceleration can improve performance significantly
- Reduce `k * 10` search multiplier in function if needed

## Test Configuration

### Modify Test Parameters
```python
# In test_category_boosted_recommendations.py

# Change recommendation count
k=10  # Default, can be modified

# Adjust policy tolerance
40 <= alignment_pct <= 60  # ±10% tolerance

# Performance target
avg_time < 1.0  # 1 second threshold
```

### Add Custom Test Cases
```python
# Add to test_policy_compliance()
{
    'name': 'Your Custom User',
    'interaction_history': [item1, item2, item3],
    'expected_category': True
}
```

## Integration with CI/CD

Add to automated testing pipeline:
```bash
# Exit code 0 = success, 1 = failure
python test_category_boosted_recommendations.py
echo $?  # Check exit code
```

---
**Generated for RecSys-HP Advanced Two-Tower Recommendation System**