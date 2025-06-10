# 🔒 Security Policy

Intel is committed to rapidly addressing security vulnerabilities affecting our
customers and providing clear guidance on the solution, impact, severity, and
mitigation.

## Security Tools and Practices

### Integrated Security Scanning

To ensure our codebase remains secure, we leverage GitHub Actions for continuous
security scanning (on pre-commit, PR and periodically) with the following tools:

- [Bandit](https://github.com/PyCQA/bandit): Static analysis tool to check Python code
- [Dependabot](https://docs.github.com/en/code-security/getting-started/dependabot-quickstart-guide): to detect security issues in dependencies

| Tool       | Pre-commit | PR-checks | Periodic |
| ---------- | -------- | --------- | -------- |
| Bandit     |         | ✅        | ✅       |
| Dependabot |          |           | ✅       |


## 🚨 Reporting a Vulnerability

Please report any security vulnerabilities in this project utilizing the
guidelines [here](https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html).

## 📢 Security Updates and Announcements

Users interested in keeping up-to-date with security announcements and updates
can:

- Follow the [GitHub repository](https://github.com/open-edge-platform/annflux) 🌐
- Check the [Releases](https://github.com/open-edge-platform/annflux/releases)
  section of our GitHub project 📦

We encourage users to report security issues and contribute to the security of
our project 🛡️. Contributions can be made in the form of code reviews, pull
requests, and constructive feedback. Refer to our
[CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

> **NOTE:** This security policy is subject to change 🔁. Users are encouraged
> to check this document periodically for updates.
