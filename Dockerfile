# STEP 1: Pull python image
FROM python:3.9.13

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /test
COPY requirements.txt /test

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;
# STEP 8: RUN COMMAND
COPY . /test
CMD ["python", "./app.py"]