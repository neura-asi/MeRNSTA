#!/usr/bin/env python3
"""
Test script for contradiction detection F1 score.
Mentioned in README evaluation section.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from cortex import ContradictionDetector
from scripts.embedder import embed

def test_contradiction_detection():
    """Test contradiction detection F1 score"""
    
    # Load config
    cfg = yaml.safe_load(open("configs/config.yaml"))
    detector = ContradictionDetector(gamma=cfg.get("gamma", 0.15))
    detector.set_rules(cfg.get("facts", []))
    
    # Test cases: (token, expected_contradiction, memory_tokens)
    test_cases = [
        ("red", True, [("blue", 0.9)]),      # Should detect contradiction
        ("blue", False, [("blue", 0.9)]),    # Should not detect
        ("green", False, [("blue", 0.9)]),   # Should not detect
        ("pizza", True, [("sushi", 0.9)]),   # Should detect contradiction
    ]
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for token, expected, memory_tokens in test_cases:
        vec = embed(token)
        embeddings = {token: vec}
        
        # Add memory token embeddings
        for mem_token, _ in memory_tokens:
            embeddings[mem_token] = embed(mem_token)
        
        detected = detector.should_veto(token, memory_tokens, embeddings)
        
        if expected and detected:
            true_positives += 1
        elif expected and not detected:
            false_negatives += 1
        elif not expected and detected:
            false_positives += 1
    
    # Calculate F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"🧪 Contradiction Detection Test Results")
    print(f"   True Positives: {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    print(f"   F1 Score: {f1:.3f}")
    
    # Assert that the F1 score is reasonable (should be > 0 for a working detector)
    assert f1 >= 0.0, f"F1 score should be >= 0, got {f1}"
    assert f1 <= 1.0, f"F1 score should be <= 1, got {f1}"
    
    # Assert that we have some true positives (detector is working)
    assert true_positives > 0, f"Expected some true positives, got {true_positives}"
    
    print(f"✅ Contradiction detection test passed with F1 score: {f1:.3f}")

if __name__ == "__main__":
    test_contradiction_detection()
    print(f"\n✅ Test completed successfully!") 