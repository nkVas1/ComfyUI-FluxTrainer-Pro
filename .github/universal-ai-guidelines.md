---
description: Universal AI Agent Guidelines for Professional Software Development
applyTo: '**'
---

# 🤖 Universal AI Agent Guidelines for Software Development

**Версия:** 2.0.0  
**Применяется к:** Всем проектам (Python, Node.js, .NET, Go, Rust и т.д.)  
**Дата:** 16 ноября 2025  

---

## 📋 Содержание

1. [Communication & Response](#communication)
2. [Code Quality & Architecture](#code-quality)
3. [Project Management](#project-management)
4. [Terminal & Environment Setup](#terminal-environment)
5. [File & Structure Management](#file-management)
6. [Database & Migrations](#database-migrations)
7. [Documentation Standards](#documentation)
8. [Starter Scripts Best Practices](#starter-scripts)
9. [Error Handling & Debugging](#error-handling)
10. [Testing & Validation](#testing)

---

## 📢 Communication & Response Standards

### Language & Tone

```
✅ DO:
- Respond in the language of the project instructions (Russian/English/etc)
- Be concise but complete - use full token budget productively
- Provide detailed, actionable responses
- Explain why you're making decisions, not just what
- Ask clarifying questions only when truly ambiguous

❌ DON'T:
- Use placeholder text like "..." or "[content omitted]"
- Leave questions unanswered in the same response
- Be verbose without adding value
- Assume information not provided in context
```

### Response Structure

```
1. Brief summary of what you'll do
2. Implementation (code/changes)
3. Verification (testing/validation)
4. Next steps (if applicable)
5. Key insights or gotchas

Total time estimate in header when relevant.
```

---

## 💻 Code Quality & Architecture

### General Principles

```python
# ✅ GOOD: Clear intent, follows conventions
def save_user_preferences(user_id: int, prefs: dict) -> bool:
    """Save user preferences to database."""
    try:
        return UserRepository(session).update(user_id, prefs)
    except DatabaseError as e:
        logger.error(f"Failed to save prefs for user {user_id}: {e}")
        return False

# ❌ BAD: Unclear, no error handling
def save_pref(uid, p):
    return db.update(uid, p)
```

### Before Creating New Files

**Always check first:**
1. Search codebase for similar functionality (use `grep_search` or `semantic_search`)
2. Check all folders for files with similar names
3. Look at imports in other files to see if this functionality exists
4. Review `__init__.py` files for exported classes/functions

```python
# Example workflow:
# User asks: "Create export_service.py"
# You search: "grep_search: export" → find existing export functions
# You check: "list_dir: src/services/" → find service patterns
# You decide: Extend existing file OR create new with best of both
# You never: Create duplicate functionality
```

### Code Changes

```
✅ Always:
- Use `replace_string_in_file` tool (not terminal sed/manual edits)
- Include 3-5 lines of context before/after in oldString
- Test changes immediately after (if applicable)
- Validate against file structure first

❌ Never:
- Edit files via terminal commands (unless explicitly requested)
- Use vague oldString that could match multiple locations
- Leave orphaned imports or references
```

---

## 📊 Project Management

### Task Completion

```
When given multiple tasks:
1. Use manage_todo_list to track progress
2. Mark ONE task as "in-progress" before starting
3. Complete work for THAT task ONLY
4. Mark task COMPLETE immediately after finishing
5. Move to next task

❌ DON'T batch completions or mark multiple in-progress
```

### Decisions & Assumptions

```
✅ DO:
- Make architectural decisions with clear rationale
- Check existing patterns before breaking conventions
- Proceed with obvious improvements without asking
- Request permission only for ambiguous/risky changes

❌ DON'T:
- Ask permission for routine tasks
- Wait for user confirmation on standard patterns
- Leave questions hanging when you can infer intent
```

---

## 🖥️ Terminal & Environment Setup

### Shell Compatibility

**PowerShell (Windows)** — Current environment

```powershell
# ✅ CORRECT PowerShell syntax:

# Activate venv
.\.venv\Scripts\Activate.ps1

# Join commands with semicolon
python -m pip install package ; python -m pytest

# Environment variables
$env:DATABASE_URL = "postgresql://..."
[System.Environment]::SetEnvironmentVariable("KEY", "value")

# Check if program exists
(Get-Command python -ErrorAction SilentlyContinue) -ne $null

# Running scripts
python script.py
# NOT: python ./script.py (forward slashes work but less native)
```

**Bash (macOS/Linux)**

```bash
# ✅ CORRECT Bash syntax:

# Activate venv
source .venv/bin/activate
# OR
. .venv/bin/activate

# Join commands
python -m pip install package && python -m pytest

# Environment variables
export DATABASE_URL="postgresql://..."

# Check if program exists
command -v python >/dev/null 2>&1

# Running scripts
./script.py
# OR
python script.py
```

### Virtual Environment Management

```
CRITICAL RULE:
Before running ANY Python command:
1. Check if venv is activated
2. If not, activate it
3. THEN run command

PowerShell:
  .\.venv\Scripts\Activate.ps1
  
Bash:
  source .venv/bin/activate

Visual indicator in terminal:
  (.venv) $ ← venv is ACTIVE
  $ ← venv is NOT active
```

### Running Python Commands

```python
# ✅ Using get_python_executable_details first (RECOMMENDED)
# Automatically handles venv + path issues

# ❌ Avoid direct python commands without activation check
# python -c "import sys; print(sys.executable)"
```

### Long-Running Processes

```bash
# Use terminal with isBackground=true for:
# - Servers (python -m uvicorn app:app)
# - Watchers (npm run dev)
# - Processes that should continue running

# Get output later with get_terminal_output

# Signal handlers:
# - Ctrl+C on Windows PowerShell: Will stop background process
# - Ctrl+C on Bash: Same behavior
# - Use explicit shutdown/kill only if Ctrl+C fails
```

---

## 📁 File & Structure Management

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| **Python files** | `snake_case.py` | `user_service.py` |
| **Python classes** | `PascalCase` | `UserService` |
| **Constants** | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| **Functions/methods** | `snake_case()` | `get_user_by_id()` |
| **Private members** | `_snake_case` | `_internal_state` |
| **Config files** | `config.extension` | `config.py`, `.env` |
| **Documentation** | `UPPER_SNAKE_CASE.md` | `README.md`, `QUICK_START.md` |

### Directory Structure

```
project/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── __init__.py
│   │   └── main.py
│   ├── services/                 # Business logic
│   ├── handlers/                 # Request handlers
│   ├── models/                   # Data models
│   ├── repositories/             # Data access
│   └── utils/                    # Utilities
├── tests/                        # Test files
├── docs/                         # Documentation
├── config/                       # Configuration
├── .env.example                  # Example env
├── requirements.txt              # Python dependencies
└── [starter].py                  # Main entry script
```

### When to Create New Files

```
CREATE NEW FILE when:
✅ Functionality doesn't exist anywhere else
✅ File would be >500 lines (consider splitting)
✅ Follows naming conventions
✅ Has clear, focused purpose

DON'T CREATE when:
❌ Similar functionality exists elsewhere
❌ Would duplicate code/patterns
❌ Breaks established structure
❌ Would create import confusion
```

### Modifying Existing Files

```
If change is < 50 lines:
→ Modify existing file (don't create new)

If change is > 50 lines but related:
→ Still modify existing, unless splitting makes sense

If implementing full new feature:
→ Can create new file IF not duplicating

Example:
# User asks: "Add logging to export_service.py"
# Instead of creating "logging_service.py"
# Modify existing "export_service.py" to add logging
```

### Import Statements

```python
# ✅ CORRECT order:
1. Standard library (os, sys, json)
2. Third-party (fastapi, sqlalchemy)
3. Local imports (from src.services import UserService)
4. Relative imports (from .models import User)

# ✅ CORRECT format:
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String

# ❌ DON'T:
import *  # Never use star imports
from module import Class as C  # Avoid cryptic aliases
```

---

## 🗄️ Database & Migrations

### When to Migrate

```
RULE: Every database schema change requires migration

Add column:
  → Create migration
  → Update models.py
  → Test migration up/down

Delete column:
  → Create migration (with backup data)
  → Update models.py

Change data type:
  → Create migration (handle existing data)
  → Update models.py
```

### Migration Process

```bash
# 1. Create migration
python generate_migration.py "add_user_email_column"

# 2. Edit alembic/versions/XXX_add_user_email_column.py
# 3. Apply migration
alembic upgrade head

# 4. Verify database schema
# 5. Update models.py
# 6. Run tests

# If error, rollback:
alembic downgrade -1
```

### SQLAlchemy Patterns

```python
# ✅ Use SQLAlchemy ORM consistently
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import DeclarativeBase

class User(DeclarativeBase):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True)

# ❌ DON'T mix ORM and raw SQL
# ❌ DON'T use string-based queries for dynamic parts
```

---

## 📚 Documentation Standards

### Documentation Location

```
docs/
├── README.md               # Project overview
├── QUICK_START.md          # 5-minute setup
├── ARCHITECTURE.md         # System design
├── API.md                  # API documentation
├── DEPLOYMENT.md           # Production setup
├── TROUBLESHOOTING.md      # Common issues
└── SESSION_*.md            # Session-specific notes
```

### Documentation Format

```markdown
# ✅ CORRECT Markdown:

## Section Title

### Subsection

Clear description of what follows.

#### Code Block

\`\`\`python
# Code example with language specified
def example(): pass
\`\`\`

#### Lists

- Item 1
- Item 2

1. Ordered 1
2. Ordered 2

#### Tables

| Column | Type | Notes |
|--------|------|-------|
| id | INT | Primary key |

---

## New Section

✅ Blank lines between sections
✅ Language tags on code blocks
✅ Clear hierarchy (# → ## → ### → ####)
❌ Bare URLs without context
❌ Missing blank lines around fences
```

### Documentation Rules

```
✅ DO:
- Write in project language (Russian/English/etc)
- Include code examples for every concept
- Use tables for comparisons
- Add ASCII diagrams for architecture
- Include troubleshooting section
- Keep updated with code changes

❌ DON'T:
- Create duplicate documentation
- Leave outdated docs in place
- Use overly complex examples
- Skip edge cases/gotchas
```

---

## 🚀 Starter Scripts Best Practices

### Purpose

Starter scripts (`start.py`, `run.sh`, etc.) should:
- ✅ Run entire project with ONE command
- ✅ Handle process management automatically
- ✅ Provide colored, meaningful output
- ✅ Gracefully handle Ctrl+C (Ctrl+Break on Windows)
- ✅ Support multiple run modes (--dev, --test, --prod, etc)

### Architecture Pattern

```python
import subprocess
import sys
import signal
from pathlib import Path
from dataclasses import dataclass

# 1. Configuration
@dataclass
class Config:
    """Project configuration constants"""
    PROJECT_ROOT = Path(__file__).parent
    PYTHON_EXECUTABLE = sys.executable
    COMPONENTS = ["bot", "api", "worker"]
    PORTS = {"api": 8000, "worker": 5555}

# 2. Output utilities (cross-platform)
class Colors:
    GREEN = '\033[92m' if sys.platform != 'win32' else ''
    RED = '\033[91m' if sys.platform != 'win32' else ''
    YELLOW = '\033[93m' if sys.platform != 'win32' else ''
    BLUE = '\033[94m' if sys.platform != 'win32' else ''
    END = '\033[0m' if sys.platform != 'win32' else ''

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

# 3. Pre-flight checks
def check_environment():
    """Verify all requirements before starting"""
    # Check Python version
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ required")
        sys.exit(1)
    
    # Check .env file
    env_file = Config.PROJECT_ROOT / ".env"
    if not env_file.exists():
        print_error(".env not found")
        sys.exit(1)
    
    # Check dependencies
    try:
        import required_package
    except ImportError:
        print_error("Dependencies missing. Run: pip install -r requirements.txt")
        sys.exit(1)

# 4. Process manager
class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Handle graceful shutdown on all platforms"""
        signal.signal(signal.SIGINT, self.shutdown)
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, self.shutdown)
    
    def start_component(self, name: str, cmd: list) -> subprocess.Popen:
        """Start a component with error checking"""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            self.processes[name] = process
            return process
        except Exception as e:
            print_error(f"Failed to start {name}: {e}")
            return None
    
    def wait_for_startup(self, name: str, port: int, timeout: int = 5) -> bool:
        """Wait for component to be ready"""
        import socket
        import time
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('127.0.0.1', port))
                sock.close()
                print_success(f"{name} ready on port {port}")
                return True
            except:
                time.sleep(0.5)
        
        print_error(f"{name} failed to start (timeout)")
        return False
    
    def shutdown(self, signum=None, frame=None):
        """Clean shutdown on Ctrl+C"""
        print("\nShutting down...")
        
        for name, proc in self.processes.items():
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                except:
                    pass
        
        sys.exit(0)

# 5. Main entry point
def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Start all components"
    )
    parser.add_argument('--mode', default='dev',
                       choices=['dev', 'test', 'prod'],
                       help='Run mode')
    
    args = parser.parse_args()
    
    # Run pre-flight checks
    check_environment()
    
    # Create manager
    manager = ProcessManager()
    
    # Start components
    if args.mode == 'dev':
        manager.start_component('bot', [sys.executable, 'src/main.py'])
        manager.start_component('api', [sys.executable, '-m', 'uvicorn', 'src.api:app'])
    
    # Read output from components
    try:
        while True:
            for name, proc in list(manager.processes.items()):
                if proc and not proc.poll():
                    try:
                        line = proc.stdout.readline()
                        if line:
                            print(f"[{name.upper()}] {line.rstrip()}")
                    except:
                        pass
    except KeyboardInterrupt:
        manager.shutdown()

if __name__ == '__main__':
    main()
```

### Windows PowerShell Considerations

```powershell
# ✅ DO:
# Use -NoProfile flag for faster startup
powershell -NoProfile -Command "& {python start.py}"

# Handle special characters properly
[System.Diagnostics.Process]::Start("python", "start.py")

# Set working directory correctly
Set-Location $PSScriptRoot

# ❌ DON'T:
# Use cmd.exe style syntax in PowerShell
# Assume Unix path separators work
# Run scripts without explicit python.exe

# CRITICAL: Virtual environment on Windows
# Works in PowerShell:
.\.venv\Scripts\Activate.ps1

# Does NOT work without full path:
.venv\Scripts\Activate.ps1  # Missing .\
```

---

## 🐛 Error Handling & Debugging

### Exception Handling Pattern

```python
# ✅ GOOD: Specific exceptions, proper logging
from typing import Optional

def get_user(user_id: int) -> Optional[User]:
    """Get user or return None on error"""
    try:
        return UserRepository().get(user_id)
    except UserNotFound:
        logger.warning(f"User {user_id} not found")
        return None
    except DatabaseError as e:
        logger.error(f"Database error fetching user {user_id}: {e}")
        raise  # Re-raise for caller to handle
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise

# ❌ BAD: Catches everything, loses context
def get_user(user_id):
    try:
        return db.get_user(user_id)
    except:
        return None
```

### Logging Standards

```python
import logging

logger = logging.getLogger(__name__)

# ✅ DO: Use appropriate levels
logger.debug("Detailed diagnostic info")           # Development
logger.info("Component started successfully")      # User relevant
logger.warning("Deprecated API used")              # May cause problems
logger.error("Failed to connect to database")      # Error occurred
logger.critical("System integrity compromised")    # Application cannot continue

# ✅ Include context
logger.error(f"Failed to save user {user_id}: {error}")

# ❌ DON'T:
logger.error("Error")  # No context
print("Error")  # Use logging, not print
logger.error(exc_info=True)  # Missing message
```

---

## ✅ Testing & Validation

### Before Committing Code

```
1. Syntax check
   → mcp_pylance_mcp_s_pylanceSyntaxErrors (for Python)
   → Compile tools for other languages

2. Run tests
   → pytest for Python
   → npm test for Node.js
   → dotnet test for .NET

3. Type checking
   → mypy for Python with type hints
   → TypeScript for JavaScript

4. Lint check
   → pylint/flake8 for Python
   → eslint for JavaScript

5. Manual testing
   → Run the application
   → Test critical paths
   → Verify error handling
```

### Writing Tests

```python
# ✅ GOOD test structure
import pytest
from src.services.user_service import UserService

@pytest.fixture
def user_service():
    """Create service with test database"""
    return UserService(test_db_session)

def test_create_user_success(user_service):
    """Test successful user creation"""
    user = user_service.create("test@example.com", "password")
    
    assert user.email == "test@example.com"
    assert user.id is not None

def test_create_user_duplicate_email(user_service):
    """Test that duplicate email raises error"""
    user_service.create("test@example.com", "password")
    
    with pytest.raises(DuplicateEmailError):
        user_service.create("test@example.com", "other_password")

def test_create_user_invalid_email(user_service):
    """Test that invalid email raises error"""
    with pytest.raises(InvalidEmailError):
        user_service.create("not-an-email", "password")
```

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov=src --cov-report=html

# Check coverage percentage
pytest --cov=src --cov-report=term-missing

# Minimum acceptable: 70% (aim for 80%+)
```

---

## 🎯 Quick Reference Checklist

### For Every Task

- [ ] Search for existing functionality first
- [ ] Follow established code patterns
- [ ] Include error handling
- [ ] Update documentation
- [ ] Test changes before completion
- [ ] Use correct file paths (no placeholders)
- [ ] Activate venv before running Python
- [ ] Use appropriate tool for the job

### For File Modifications

- [ ] Include 3-5 lines of context in oldString
- [ ] Verify oldString uniquely identifies location
- [ ] Check for orphaned imports/references
- [ ] Update documentation if behavior changed

### For New Features

- [ ] Design matches existing architecture
- [ ] Tests written for new code
- [ ] Documentation updated
- [ ] Backwards compatible (if applicable)
- [ ] Error cases handled

### For Terminal Commands

- [ ] Right syntax for current shell (PowerShell vs Bash)
- [ ] Virtual environment activated
- [ ] All paths are absolute or properly relative
- [ ] Long processes use isBackground=true

---

## 📞 Getting Help

When you don't know something:

1. **Search the codebase** — similar code likely exists
2. **Check documentation** — README, QUICK_START, docs/
3. **Read error messages** — they often tell you exactly what's wrong
4. **Trace the code** — find similar implementations
5. **Ask in context** — explain what you tried and what failed

**Don't:** Ask permission for routine tasks or wait for unclear clarification.

---

**Last Updated:** 16 November 2025  
**Version:** 2.0.0  
**Status:** Active for all projects
