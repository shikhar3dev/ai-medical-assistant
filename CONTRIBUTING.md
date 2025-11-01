# Contributing to Privacy-Preserving Federated Learning

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ai-disease-prediction.git
   cd ai-disease-prediction
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name.
   ```

## Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

2. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Run tests** to ensure everything works:
   ```bash
   pytest tests/ -v
   ```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- **Line length**: 100 characters maximum
- **Formatting**: Use `black` for automatic formatting
- **Linting**: Use `flake8` for linting
- **Type hints**: Use type hints where possible

### Format your code:
```bash
black .
flake8 .
mypy .
```

## Making Changes

### 1. Write Clean Code

- Follow existing code patterns
- Add docstrings to all functions and classes
- Use meaningful variable names
- Keep functions focused and small

### 2. Add Tests

- Write tests for new features
- Ensure all tests pass: `pytest tests/ -v`
- Aim for >80% code coverage

### 3. Update Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update CHANGELOG.md

### 4. Commit Messages

Use clear, descriptive commit messages:

```
feat: Add SHAP waterfall plot visualization
fix: Resolve gradient clipping bug in DP trainer
docs: Update installation instructions
test: Add tests for LIME explainer
refactor: Simplify data partitioning logic
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests**:
   ```bash
   pytest tests/ -v --cov
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Screenshots (if UI changes)
   - Test results

5. **Address review comments** promptly

## Areas for Contribution

### High Priority

- **Real Hospital Data Integration**: Support for real medical datasets
- **Advanced Privacy Mechanisms**: Homomorphic encryption, secure multi-party computation
- **Model Architectures**: Support for transformers, attention mechanisms
- **Deployment Tools**: Docker containers, Kubernetes configs
- **Monitoring**: Real-time training monitoring, alerting

### Medium Priority

- **Additional Datasets**: Support for more medical datasets
- **Visualization**: Enhanced plots and dashboards
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Integration tests, performance tests
- **Optimization**: Training speed, memory efficiency

### Good First Issues

- **Documentation improvements**: Fix typos, clarify instructions
- **Code cleanup**: Remove duplicates, improve naming
- **Test coverage**: Add missing tests
- **Examples**: Create example notebooks
- **Bug fixes**: Fix reported issues

## Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: OS, Python version, package versions
6. **Logs**: Relevant error messages or logs

## Suggesting Features

When suggesting features, include:

1. **Use case**: Why is this feature needed?
2. **Description**: What should the feature do?
3. **Examples**: How would it be used?
4. **Alternatives**: Other approaches considered

## Code Review Guidelines

When reviewing code:

- Be respectful and constructive
- Focus on the code, not the person
- Explain your reasoning
- Suggest improvements, don't just criticize
- Approve when ready, request changes if needed

## Community Guidelines

- Be welcoming and inclusive
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the project
- Show empathy towards others

## Questions?

- Open an issue for questions
- Join discussions on GitHub
- Check existing issues and PRs first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
