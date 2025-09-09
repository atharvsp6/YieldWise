# Contributing to YieldWise

Thank you for your interest in contributing to YieldWise! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork the Repository
- Click the "Fork" button on the GitHub repository page
- Clone your forked repository to your local machine

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes
- Write clean, readable code
- Follow the existing code style
- Add comments for complex logic
- Update documentation if needed

### 4. Test Your Changes
- Test the functionality thoroughly
- Ensure all existing tests pass
- Add new tests for new features

### 5. Commit Your Changes
```bash
git add .
git commit -m "Add: brief description of your changes"
```

### 6. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request
- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your feature branch
- Provide a clear description of your changes

## 📋 Development Setup

### Prerequisites
- Python 3.10+
- Git
- Virtual environment (venv)

### Setup Steps
1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies
4. Set up environment variables
5. Run the application

```bash
git clone https://github.com/atharvsp6/YieldWise.git
cd YieldWise
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
python app.py
```

## 🎯 Areas for Contribution

### 🐛 Bug Fixes
- Fix existing issues
- Improve error handling
- Optimize performance

### ✨ New Features
- Additional crop types
- New disease detection models
- Enhanced financial analysis
- Mobile responsiveness improvements
- API endpoints
- Database integration

### 📚 Documentation
- Improve README
- Add code comments
- Create tutorials
- Update API documentation

### 🧪 Testing
- Unit tests
- Integration tests
- End-to-end tests
- Performance tests

### 🎨 UI/UX
- Design improvements
- Accessibility enhancements
- Mobile optimization
- User experience improvements

## 📝 Code Style Guidelines

### Python
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings for functions and classes
- Keep functions small and focused

### JavaScript
- Use consistent indentation (2 spaces)
- Use meaningful variable names
- Comment complex logic
- Follow modern ES6+ practices

### HTML/CSS
- Use semantic HTML
- Follow BEM methodology for CSS
- Ensure accessibility
- Mobile-first responsive design

## 🐛 Reporting Issues

### Before Reporting
1. Check if the issue already exists
2. Try the latest version
3. Reproduce the issue

### Issue Template
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. macOS, Windows, Linux]
- Python version: [e.g. 3.10.0]
- Browser: [e.g. Chrome, Firefox, Safari]

**Additional context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Requesting
1. Check if the feature already exists
2. Consider if it aligns with the project goals
3. Think about implementation complexity

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

## 🏷️ Pull Request Guidelines

### PR Template
```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📞 Getting Help

- Create an issue for questions
- Join our community discussions
- Contact maintainers directly

## 📄 License

By contributing to YieldWise, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to YieldWise! 🌾
