# 🧠 MeRNSTA Quick Start Guide

## 🚀 **Super Quick Start** *(Recommended)*

### **One Command Setup**
```bash
# Clone and setup
git clone https://github.com/icedmoca/mernsta.git
cd mernsta
pip install -r requirements.txt

# Start everything (Ollama + MeRNSTA)
./start_mernsta_with_ollama.sh
```

### **Manual Setup**
```bash
# 1. Start Ollama (custom build with tokenizer/detokenizer)
./scripts/start_ollama.sh start

# 2. Start MeRNSTA
python main.py run
```

## 🔧 **Ollama Setup** *(Required First)*

This repository uses a **custom Ollama build** with enhanced tokenizer/detokenizer support.

### **Quick Ollama Commands**
```bash
./scripts/start_ollama.sh start    # Start Ollama
./scripts/start_ollama.sh status   # Check status
./scripts/start_ollama.sh stop     # Stop Ollama
./scripts/start_ollama.sh logs     # View logs
```

### **Manual Ollama Start**
```bash
cd external/ollama
./ollama serve
```

**⚠️ Important:** Ollama must be running before starting MeRNSTA!

## 🎯 **Running MeRNSTA**

### **Full AGI Mode** *(Recommended)*
```bash
python main.py run
```

### **Simple Interactive Mode**
```bash
python main.py interactive
```

### **Web Interface**
```bash
python main.py web
```

### **CLI Shell**
```bash
python main.py cli
```

### **API Server**
```bash
python main.py api
```

## 📊 **Access Points**

- **💬 Web Chat:** http://localhost:8000/chat
- **🔌 REST API:** http://localhost:8001/docs
- **📊 Health Status:** http://localhost:8000/health

## 🧪 **Example Interactions**

1. **Basic Statement**: "I love programming in Python"
2. **Theory of Mind**: "Alice thinks AI will take over the world"  
3. **Create Contradiction**: "I hate programming actually"
4. **Query Memory**: "What do I think about programming?"
5. **Introspect**: "/introspect"
6. **Check Clarifications**: "/clarify"

## 🎮 **Special Commands**

- `/introspect` - View cognitive state & self-analysis
- `/clarify` - Show pending clarification requests  
- `/tune` - Display autonomous memory tuning status
- `/perspectives` - Show Theory of Mind tracking

## ✅ **Features Working**

✅ **Causal & Temporal Linkage** - Tracks belief evolution  
✅ **Dialogue Clarification** - Auto-generates clarifying questions  
✅ **Autonomous Memory Tuning** - Self-adjusts parameters  
✅ **Theory of Mind** - Multi-perspective belief tracking  
✅ **Recursive Self-Inspection** - Cognitive state analysis  
✅ **Contradiction Detection** - Real-time conflict resolution  
✅ **Volatility Tracking** - Belief instability monitoring  
✅ **Custom Ollama Integration** - Enhanced tokenizer/detokenizer support

## 🔍 **Troubleshooting**

### **Ollama Issues**
```bash
# Check Ollama setup
python utils/ollama_checker.py --validate

# Show detailed status
python utils/ollama_checker.py --detailed

# Show setup instructions
python utils/ollama_checker.py --instructions
```

### **Common Problems**
- **"Ollama not running"** → Run `./scripts/start_ollama.sh start`
- **"Tokenizer endpoints not responding"** → Check Ollama logs with `./scripts/start_ollama.sh logs`
- **"Binary not found"** → Ensure you have the custom Ollama build in `external/ollama/`

## 📚 **Need Help?**

- Type `help` in the interactive session
- Check the main README.md for detailed documentation
- Use `./start_mernsta_with_ollama.sh --help` for startup options
