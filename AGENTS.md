This document guides how code and notebooks are structured for CoreLab, including how AI agents should edit files and how humans should work locally.

## Project structure
- `/aws` input from official AWS sources such as github
- `/labs` notebooks and other resources per lab

## Coding conventions
- Python is used for all code
- Code is put in python files, notebooks are for interacting with code
- Each notebook gets a auto reload section at the top
- Plumbing code is seperated from code/domain code for maximum readability and portability

## Tools used
- git for version control
- GitHub for collaboration and hosting
- Jupyter Notebook for interactive coding and documentation
- Python for scripting and automation
- AWS for cloud infrastructure and services
- uv for python project management
