# The ResearchPy Philosophy: A Research-First Development Model

## Core Principles
1. **Implementation lives in `utils.py`. Orchestration lives in scripts.**
2. **Fail fast. No fallbacks. Explicit is better than implicit.**
3. **Script legibility is paramount - anyone should understand the flow at a glance.**

## The Fail-Fast Philosophy

**Silent failures waste GPU hours and corrupt experiments.**

### What NOT to do:
```python
# ❌ BAD: Silent fallback that hides critical errors
if config is None or 'randomwalk' not in config:
    print("Warning: randomwalk evaluation requires config")  # Just a warning!
    cities_df = None  # Silently continues with broken state
else:
    cities_df = pd.read_csv(config['randomwalk']['cities_csv'])

# Later...
if cities_df is not None:  # Silently skips evaluation
    # do evaluation
else:
    return fake_metrics  # Returns zeros, pretending to work!
```

This code ran for HOURS returning fake 0.0 metrics because config wasn't passed. No crash = wasted compute.

### What to do instead:
```python
# ✅ GOOD: Immediate, loud failure
if config is None or 'randomwalk' not in config:
    raise ValueError("FATAL: randomwalk evaluation requires config with randomwalk section")

cities_df = pd.read_csv(config['randomwalk']['cities_csv'])  # No fallback path
# Now cities_df is GUARANTEED to exist
```

### Why this matters in research:
1. **GPU time is expensive** - Silent failures waste thousands of dollars
2. **Wrong metrics mislead** - You think your model is training when it's not
3. **Debugging is faster** - Crash immediately at the problem, not 10 functions later
4. **Reproducibility** - No hidden fallback behaviors to document

### Rules:
- **Required parameters**: Crash if missing, no defaults
- **File operations**: Crash if file doesn't exist, don't create fallbacks
- **Config validation**: Validate everything upfront, crash on first issue
- **Type checking**: Assert types explicitly, don't coerce
- **Dependencies**: If something is required, make it REQUIRED

Remember: It's better to crash in 1 second than to run for 10 hours with wrong behavior.

## Real Examples from This Codebase

### Example 1: Separating Implementation from Orchestration

**`src/utils.py`** - The HOW (implementation):
```python
def get_model(config):
    """Initialize model from config."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    tokenizer.padding_side = 'right'
    
    # Create model config
    model_config = Qwen2Config(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        # ... all parameters ...
    )
    
    # Initialize model
    model = Qwen2ForCausalLM(model_config)
    model.apply(lambda m: init_weights(m, config['model']['init_scale']))
    
    # Attach tokenizer as convention
    model.tokenizer = tokenizer
    return model

def parse_location(text):
    """Parse location from text like 'loc(c_1234)=567,890'."""
    match = re.search(r'=(-?\d+),(-?\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r'(-?\d+),(-?\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None
```

**`src/training/train.py`** - The WHAT/WHEN (orchestration):
```python
def main():
    # Parse arguments
    args = parser.parse_args()
    
    # Load and validate config
    config = preprocess_config(config)
    
    # Initialize experiment directory
    exp_dir = init_experiment_directory(config['exp_dir'], overwrite)
    
    # Get datasets and model
    train_dataset, eval_dataset, tokenizer = get_dataset(config)
    model = get_model(config)
    
    # Train
    trainer = Trainer(model=model, args=training_args, ...)
    trainer.train()
```

Notice how the script reads like a story - each line is a meaningful step.

### Example 2: Fail Fast with No Fallbacks

**Before** (with fallbacks that hide errors):
```python
# ❌ BAD: Fallback hides missing config
tokenizer_path = config.get('tokenizer_path', 'outputs/tokenizer')
learning_rate = config.get('learning_rate', 1e-4)
```

**After** (fail fast):
```python
# ✅ GOOD: From our preprocess_config in utils.py
def preprocess_config(config):
    required_fields = {
        'task_type': ('str', ['location', 'distance'], "Task type must be 'location' or 'distance'"),
        'tokenizer_path': ('str', None, "Path to tokenizer"),
        'training.learning_rate': ('float', None, "Learning rate"),
        # ... all required fields ...
    }
    
    for field_path, (expected_type, valid_values, description) in required_fields.items():
        # Check field exists (no fallback!)
        if value is None:
            raise ValueError(f"Missing required field: {field_path} - {description}")
```

### Example 3: Trust the Framework

**Before** (reimplementing what HuggingFace provides):
```python
# ❌ BAD: 26 lines we had for custom optimizer/scheduler
def get_optimizer_and_scheduler(trainer):
    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=int(total_training_steps)
    )
    return optimizer, scheduler

trainer.optimizer, trainer.lr_scheduler = get_optimizer_and_scheduler(trainer)
```

**After** (using framework features):
```python
# ✅ GOOD: Let HuggingFace handle it
training_args = TrainingArguments(
    lr_scheduler_type="linear",
    warmup_steps=config['training']['warmup_steps'],
    learning_rate=config['training']['learning_rate'],
)
trainer = Trainer(args=training_args, ...)  # That's it!
```

### Example 4: Group Related Functions

**In our `utils.py`**:
```python
# ✅ GOOD: Task-specific parsing functions are neighbors
def parse_location(text):
    """Parse location from text like 'loc(c_1234)=567,890'."""
    match = re.search(r'=(-?\d+),(-?\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def parse_distance(text):
    """Parse distance from text like 'dist(c_1234,c_5678)=901'."""
    match = re.search(r'=(\d+)', text)
    if match:
        return int(match.group(1))
    return None
```

### Example 5: Everything in Main

**Before** (module-level execution):
```python
# ❌ BAD: This was at module level in our old train.py
load_dotenv()
parser = argparse.ArgumentParser()
args = parser.parse_args()
config = yaml.safe_load(open(args.config_path))
exp_dir = init_experiment_directory(config['exp_dir'])

def main():
    # training code
```

**After** (clean module):
```python
# ✅ GOOD: Current train.py - everything in main()
def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path))
    exp_dir = init_experiment_directory(config['exp_dir'])
    # training code

if __name__ == "__main__":
    main()
```

### Example 6: Don't Over-Modularize

**What we avoided** (over-engineered):
```python
# ❌ BAD: Too many tiny functions
def _get_tokenizer_path(config):
    return config['tokenizer_path']

def _setup_tokenizer(path):
    return AutoTokenizer.from_pretrained(path)

def _load_tokenizer(config):
    return _setup_tokenizer(_get_tokenizer_path(config))
```

**What we have** (clear and chunky):
```python
# ✅ GOOD: Our actual get_model() - one complete operation
def get_model(config):
    """Initialize model from config."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    tokenizer.padding_side = 'right'
    
    # Create model config  
    model_config = Qwen2Config(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        # ... all 10+ parameters explicitly set ...
    )
    
    # Initialize and setup model
    model = Qwen2ForCausalLM(model_config)
    model.apply(lambda m: init_weights(m, config['model']['init_scale']))
    model.tokenizer = tokenizer
    
    return model
```

This 20+ line function is perfectly fine - it does one complete, meaningful operation.

## Quick Test

Ask yourself:
- "Could another experiment use this function unchanged?" → `utils.py`
- "Is this specific to THIS experiment's flow?" → script  
- "Is this HOW to do something?" → `utils.py`
- "Is this WHEN/WHETHER to do something?" → script
- "Am I fighting the framework?" → Use framework features

**The Golden Rule**: Implementation is HOW. Orchestration is WHAT/WHEN/WHETHER.