# Kilo Agent Configuration

This file documents agent configurations and workflow practices for this project.

## Available Agents

### fullstack-architect (Global)
A visionary full-stack architect who thinks big and builds ambitious, scalable systems. Specializes in:
- **React frontends**: Component-driven architecture, atomic design, advanced state management, real-time features
- **Python backends**: Async-first FastAPI, clean architecture, horizontal scaling, observability
- **System thinking**: Always considers scale, integration, extensibility, and long-term maintainability

### code-reviewer
Senior software engineer conducting thorough code reviews (quality, security, performance, maintainability).

### code-simplifier
Expert refactoring specialist for cleaner, more maintainable code.

### test-engineer
QA specialist for comprehensive tests and improved code coverage.

## Project-Specific Guidelines

- Python code follows existing patterns in `src/` directory
- Async-first design with proper error handling and retry logic
- Weaviate integration with embedded vectors for server-side permission issues
- Thread-safe model initialization with locks for concurrent access