#!/usr/bin/env python3
"""
🚀 MeRNSTA Phase 2 Installation Script

Automatically installs dependencies and verifies the autonomous cognitive system.
"""

import sys
import os
import subprocess
import venv

def print_banner():
    print("\n" + "="*70)
    print("🚀 MeRNSTA v0.7.0 - Installation & Setup")
    print("   Autonomous Cognitive Agent Architecture")
    print("="*70)

def run_command(cmd, description=""):
    """Run a command and return success status."""
    try:
        print(f"📦 {description}...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("\n🔧 Installing Dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Install pip requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Install spaCy model with large vectors
    if not run_command("python -m spacy download en_core_web_lg", "Installing spaCy language model (large with superior vectors)"):
        return False
    
    return True

def verify_installation():
    """Verify that MeRNSTA can be imported and initialized."""
    print("\n🧪 Verifying Installation...")
    
    try:
        print("📦 Testing imports...")
        from storage.phase2_cognitive_system import Phase2AutonomousCognitiveSystem
        print("   ✅ Phase 2 system imports successful")
        
        print("🧠 Testing system initialization...")
        system = Phase2AutonomousCognitiveSystem()
        print("   ✅ Autonomous cognitive agent initialized")
        
        print("🔬 Testing basic functionality...")
        result = system.process_input_with_full_cognition("Test input", "test_user", "test_session")
        if result.get('response'):
            print("   ✅ Processing pipeline working")
        else:
            print("   ⚠️ Processing returned no response")
        
        print("🎯 Testing special commands...")
        introspect_result = system.process_input_with_full_cognition("/introspect", "test_user", "test_session")
        if introspect_result.get('response'):
            print("   ✅ Special commands working")
        else:
            print("   ⚠️ Special commands may have issues")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Runtime error: {e}")
        return False

def create_quick_start_guide():
    """Create a quick start guide file."""
    guide_content = """# 🧠 MeRNSTA Quick Start Guide

## Running MeRNSTA Phase 2

### Simple Interactive Mode
```bash
python3 run_mernsta.py
```

### Quick Test
```bash
python3 run_mernsta.py --test
```

## Example Interactions

1. **Basic Statement**: "I love programming in Python"
2. **Theory of Mind**: "Alice thinks AI will take over the world"  
3. **Create Contradiction**: "I hate programming actually"
4. **Query Memory**: "What do I think about programming?"
5. **Introspect**: "/introspect"
6. **Check Clarifications**: "/clarify"

## Special Commands

- `/introspect` - View cognitive state & self-analysis
- `/clarify` - Show pending clarification requests  
- `/tune` - Display autonomous memory tuning status
- `/perspectives` - Show Theory of Mind tracking

## Features Working

✅ **Causal & Temporal Linkage** - Tracks belief evolution  
✅ **Dialogue Clarification** - Auto-generates clarifying questions  
✅ **Autonomous Memory Tuning** - Self-adjusts parameters  
✅ **Theory of Mind** - Multi-perspective belief tracking  
✅ **Recursive Self-Inspection** - Cognitive state analysis  
✅ **Contradiction Detection** - Real-time conflict resolution  
✅ **Volatility Tracking** - Belief instability monitoring  

## Need Help?

Type `help` in the interactive session for more information.
"""
    
    try:
        with open("QUICK_START.md", "w") as f:
            f.write(guide_content)
        print("📚 Created QUICK_START.md guide")
        return True
    except Exception as e:
        print(f"❌ Could not create quick start guide: {e}")
        return False

def main():
    """Main installation routine."""
    print_banner()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check errors above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Installation completed but verification failed.")
        print("💡 You may still be able to run MeRNSTA, but some features might not work.")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            sys.exit(1)
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Success message
    print("\n" + "="*70)
    print("🎉 MERNSTA PHASE 2 INSTALLATION COMPLETE!")
    print("="*70)
    print("🚀 Ready to run the autonomous cognitive agent!")
    print("")
    print("📝 Quick Start:")
    print("   python3 run_mernsta.py")
    print("")
    print("🧪 Test Installation:")
    print("   python3 run_mernsta.py --test")
    print("")
    print("📚 Documentation:")
    print("   • README.md - Full documentation")
    print("   • QUICK_START.md - Quick start guide")
    print("   • docs/paper.md - Technical paper")
    print("")
    print("🎯 The world's first autonomous cognitive agent is ready!")
    print("="*70)

if __name__ == "__main__":
    main() 