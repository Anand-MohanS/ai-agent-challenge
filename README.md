# AI Agent Challenge - Bank Statement Parser Generator

An "Agent-as-Coder" system that autonomously generates Python parsers for bank statement PDFs using LangGraph and follows a plan → code → test → self-fix loop.

## Quick Start (5 Steps)

1. **Clone and Setup Environment**
   ```bash
   git clone <repository-url>
   cd ai-agent-challenge
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Agent for ICICI Bank**
   ```bash
   python agent.py --target icici
   ```

4. **Test Generated Parser**
   ```bash
   python -m pytest test_agent.py -v
   ```

5. **Use Generated Parser**
   ```bash
   python custom_parser/icici_parser.py "data/icici/icici sample.pdf"
   ```

## Architecture Overview

The system implements an autonomous coding agent using LangGraph's stateful workflow management. The architecture consists of five main nodes connected by conditional edges:

**Core Workflow:**
- **PDF Analysis Node**: Extracts table structure and sample data using pdfplumber
- **Parser Generation Node**: Creates custom Python parser code based on discovered patterns
- **Testing Node**: Validates parser output against expected DataFrame schema
- **Fix Node**: Analyzes test failures and regenerates improved code (max 3 attempts)
- **Finalization Node**: Completes the workflow and saves the final parser

**State Management**: Uses MessagesState to maintain conversation history, intermediate results, and error tracking throughout the autonomous debugging process. The agent demonstrates self-correction capabilities by analyzing test failures and iteratively improving the generated code until it passes validation or reaches the maximum retry limit.

##  Project Structure

```
ai-agent-challenge/
├── agent.py              # Main agent implementation
├── test_agent.py         # Pytest validation suite
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/
│   └── icici/
│       └── icici sample.pdf  # Sample bank statement
└── custom_parser/       # Generated parsers (created by agent)
    └── icici_parser.py  # Auto-generated ICICI parser
```

##  Technical Features

- **Autonomous Code Generation**: Self-debugging loops with error correction
- **LangGraph Workflow**: Stateful graph-based agent orchestration  
- **PDF Table Extraction**: Advanced parsing using pdfplumber
- **Type Hints & Documentation**: Clean, maintainable code generation
- **Pytest Integration**: Automated validation with DataFrame.equals comparison
- **Modular Architecture**: Clear separation of concerns across workflow nodes

##  Testing

The system includes comprehensive pytest validation:

```bash
# Run all tests
python -m pytest test_agent.py -v

# Test specific functionality
python -m pytest test_agent.py::test_icici_parser_output -v
```

##  Sample Output

The generated parser extracts structured data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| Date | string | Transaction date |
| Description | string | Transaction description |
| Debit Amt | float | Debit amount (if applicable) |
| Credit Amt | float | Credit amount (if applicable) |
| Balance | float | Account balance after transaction |

##  Agent Workflow

1. **Plan**: Analyze PDF structure and identify table patterns
2. **Code**: Generate custom parser using discovered patterns
3. **Test**: Validate parser output against expected schema
4. **Self-Fix**: Debug and improve code if tests fail (up to 3 attempts)
5. **Complete**: Finalize working parser or report failure


## 📝 Usage Example

```python
from custom_parser.icici_parser import parse
import pandas as pd

# Parse bank statement
df = parse("data/icici/icici sample.pdf")

# Display results
print(f"Extracted {len(df)} transactions")
print(df.head())
```
