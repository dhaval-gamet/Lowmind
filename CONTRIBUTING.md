# Contributing to Custom Deep Learning Framework

We're thrilled you're interested in contributing to our project! Your help is invaluable in making this framework better. This guide will walk you through the process of contributing, from reporting issues to submitting pull requests.

## 🐛 Reporting Issues

If you encounter a bug, have a feature request, or want to suggest an improvement, please open a new issue.

**Before submitting a new issue:**
* **Search existing issues:** Check if a similar issue has already been reported. This helps avoid duplicates.
* **Be clear and detailed:** Provide as much information as possible.
    * **For bugs:** Describe the expected behavior and what actually happened. Include the steps to reproduce the bug and any error messages you received.
    * **For feature requests:** Clearly explain the problem you're trying to solve and how the new feature would help.

## 🤝 Submitting a Pull Request (PR)

Pull requests are the best way to contribute code. We appreciate clean code, descriptive commits, and a clear explanation of your changes.

**Steps to submit a PR:**

1.  **Fork the Repository:** Create a copy of the repository on your GitHub account.
2.  **Clone Your Fork:** Clone your forked repository to your local machine.
    ```bash
    git clone [https://github.com/dhaval-gamet/lowmind.git](https://github.com/dhaval-gamet/lowmind.git)
    lowmind
    ```
3.  **Create a New Branch:** Always work on a new branch, giving it a descriptive name (e.g., `feature/add-tanh-layer` or `fix/conv-padding-bug`).
    ```bash
    git checkout -b your-branch-name
    ```
4.  **Make Your Changes:** Implement your bug fix or new feature.
5.  **Test Your Changes:** Ensure your code works as expected and doesn't break existing functionality. Run the example scripts (`main.py`) to verify.
6.  **Commit Your Changes:** Write clear, concise commit messages. A good commit message explains *what* you did and *why*.
    ```bash
    git add .
    git commit -m "feat: Add Tanh activation function"
    ```
7.  **Push to Your Fork:** Push your new branch to your forked repository on GitHub.
    ```bash
    git push origin your-branch-name
    ```
8.  **Open a Pull Request:** Go to your repository on GitHub, and you'll see a button to "Compare & pull request." Write a clear title and description for your PR.

## 🔖 **Issue Labels**

We use labels to help categorize issues and make it easier for contributors to find tasks. Look for issues with the following labels:

* `good first issue`: Great for new contributors. These are usually small, self-contained tasks.
* `help wanted`: Issues that are ready for a community member to pick up.
* `bug`: Something is not working as it should.
* `enhancement`: A new feature or improvement to existing functionality.

Thank you for your interest in contributing! We look forward to your pull requests.
