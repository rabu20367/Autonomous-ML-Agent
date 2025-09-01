# Contributing to Autonomous ML Agent

Thank you for your interest in contributing to the Autonomous ML Agent project! We welcome contributions from the community and appreciate your help in making this project better.

## 🤝 How to Contribute

### 1. **Fork the Repository**
- Click the "Fork" button on the GitHub repository page
- Clone your forked repository to your local machine

### 2. **Set Up Development Environment**
```bash
git clone https://github.com/rabu20367/autonomous-ml-agent.git
cd autonomous-ml-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. **Create a Branch**
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 4. **Make Your Changes**
- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style

### 5. **Test Your Changes**
```bash
# Run tests
python -m pytest tests/

# Run linting
flake8 autonomous_ml/
black autonomous_ml/

# Run type checking
mypy autonomous_ml/
```

### 6. **Commit Your Changes**
```bash
git add .
git commit -m "Add: Brief description of your changes"
```

### 7. **Push and Create Pull Request**
```bash
git push origin feature/your-feature-name
```
Then create a Pull Request on GitHub.

## 📋 Types of Contributions

### 🐛 **Bug Reports**
- Use the GitHub issue template
- Provide clear steps to reproduce
- Include system information and error messages

### ✨ **Feature Requests**
- Describe the feature clearly
- Explain the use case and benefits
- Consider implementation complexity

### 🔧 **Code Contributions**
- Bug fixes
- New features
- Performance improvements
- Documentation updates
- Test coverage improvements

### 📚 **Documentation**
- README improvements
- Code comments
- API documentation
- Tutorials and examples

## 🎯 Areas for Contribution

### **High Priority**
- [ ] Additional ML algorithms (SVM, XGBoost, LightGBM)
- [ ] Advanced preprocessing techniques
- [ ] Model interpretability features
- [ ] Performance optimizations
- [ ] Web interface improvements

### **Medium Priority**
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Additional dataset examples
- [ ] API documentation
- [ ] Unit test coverage

### **Low Priority**
- [ ] Multi-language support
- [ ] Advanced visualization features
- [ ] Integration with cloud platforms
- [ ] Mobile app interface

## 📝 Code Style Guidelines

### **Python Code**
- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused
- Use meaningful variable names

### **Example:**
```python
def train_model(
    X: pd.DataFrame, 
    y: pd.Series, 
    model_name: str = "random_forest"
) -> Dict[str, Any]:
    """
    Train a machine learning model on the given data.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_name: Name of the model to train
        
    Returns:
        Dictionary containing model and performance metrics
        
    Raises:
        ValueError: If model_name is not supported
    """
    # Implementation here
    pass
```

### **Commit Messages**
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for test additions/changes
- `chore:` for maintenance tasks

Examples:
```
feat: add support for XGBoost algorithm
fix: resolve memory leak in data preprocessing
docs: update README with installation instructions
```

## 🧪 Testing Guidelines

### **Write Tests For**
- New functionality
- Bug fixes
- Edge cases
- Error conditions

### **Test Structure**
```python
def test_model_training():
    """Test that model training works correctly."""
    # Arrange
    X, y = create_test_data()
    
    # Act
    result = train_model(X, y)
    
    # Assert
    assert result['accuracy'] > 0.8
    assert 'model' in result
```

### **Run Tests**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=autonomous_ml

# Run with verbose output
pytest -v
```

## 📖 Documentation Standards

### **Code Documentation**
- Use Google-style docstrings
- Include type hints
- Document all public functions and classes
- Provide usage examples where helpful

### **README Updates**
- Keep installation instructions current
- Update feature lists
- Add new examples
- Include screenshots for UI changes

## 🔍 Review Process

### **Pull Request Requirements**
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Clear description of changes

### **Review Criteria**
- Code quality and readability
- Test coverage
- Performance impact
- Security considerations
- Documentation completeness

## 🏷️ Release Process

### **Version Numbering**
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### **Release Checklist**
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release

## 💬 Community Guidelines

### **Be Respectful**
- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community

### **Be Collaborative**
- Help others learn and grow
- Share knowledge and resources
- Work together toward common goals
- Give credit where credit is due

### **Be Professional**
- Keep discussions focused on the project
- Avoid personal attacks or harassment
- Follow the project's code of conduct
- Report inappropriate behavior

## 🆘 Getting Help

### **Questions and Support**
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: atm.hasibur.rashid20367@gmail.com for private matters

### **Resources**
- [Python Documentation](https://docs.python.org/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)

## 🎉 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributor graph

Thank you for contributing to the Autonomous ML Agent project! 🚀
