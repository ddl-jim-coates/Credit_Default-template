# Credit Default Model Template
This template is to be used as the starting point for all credit scoring and credit default models.  It includes the following:
- Domino Environment with approved libraries and configurations
- Approved Domino Data Sources for Credit scoring
- Model Development Tasks/Goals
- Project file structure
- Sample Notebooks, Training, and Application Scripts.
- Recommended Hardware Tier
- Starting Tags

## Project File Structure Structure
Use the Project Template directory structure to organize files for this project.
- apps
- documentation
- models
- notebooks
- results
- scripts
- visualizations

## Domino Environment Definition
[Domino Environment](R115374714469.dkr.ecr.us-west-2.amazonaws.com/cloud-sedemo/environment:6760751fa89a19743d395951-5)

FROM
```
quay.io/domino/domino-standard-environment:ubuntu22-py3.10-r4.4-domino6.0-standard
```
DOCKER INSTRUCTIONS
```
USER root
RUN pip install --upgrade pandas seaborn
RUN pip install h2o streamlit eli5
USER ubuntu
```
PLUGABLE IDEs
```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/opt/domino/workspaces/jupyter/start" ]
  supportedFileExtensions: [ ".ipynb" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [  "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
  title: "vscode"
  iconUrl: "/assets/images/workspace-logos/vscode.svg"
  start: [ "/opt/domino/workspaces/vscode/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
rstudio:
  title: "RStudio"
  iconUrl: "/assets/images/workspace-logos/Rstudio.svg"
  start: [ "/opt/domino/workspaces/rstudio/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```