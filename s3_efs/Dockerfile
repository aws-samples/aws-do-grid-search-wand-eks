FROM ubuntu:20.04

RUN apt-get update && apt-get install -y curl unzip

#Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install

#Remove aws cli artifacts after installation
RUN rm -rf aws && rm awscliv2.zip

ADD data-prep.sh /
RUN chmod +x /data-prep.sh

