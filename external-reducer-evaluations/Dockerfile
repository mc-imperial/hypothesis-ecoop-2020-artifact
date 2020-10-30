FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python2.7-dev build-essential curl apt-transport-https apt-utils software-properties-common bzip2 libc6-dev-i386 make git bash

ADD build-files/install-compilers.sh /root/

RUN /root/install-compilers.sh

# Ensure that we always use UTF-8 and with Canadian English locale
RUN apt-get install -y locales && locale-gen en_GB.UTF-8

COPY build-files/default_locale /etc/default/locale
RUN chmod 0755 /etc/default/locale

ENV LC_ALL=en_GB.UTF-8
ENV LANG=en_GB.UTF-8
ENV LANGUAGE=en_GB.UTF-8

ADD build-files/apt-installs.sh /root/
RUN /root/apt-installs.sh

ADD build-files/llvm-install.sh /root/
RUN /root/llvm-install.sh

ADD build-files/*.sh /root/
RUN /root/install.sh

ADD requirements.txt /root/
RUN pip install --upgrade pip && pip install -r /root/requirements.txt

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get install -y libbsd-dev
