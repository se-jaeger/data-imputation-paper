
FROM python:3.8.8

RUN pip install tensorflow typer autokeras optuna pandas scikit-learn imgaug imagecorruptions

# Add code
ADD . /workdir

WORKDIR /workdir/src/jenga
RUN python setup.py develop

WORKDIR /workdir
RUN python setup.py develop

WORKDIR /workdir/scripts

ENTRYPOINT ["python", "run-experiment.py"]
