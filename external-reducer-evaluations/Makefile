STATEFILES=.working
DOCKER_BUILT=$(STATEFILES)/Dockerfile.built
STATEDIR_BUILT=$(STATEFILES)/exists

DOCKER_TAG=reducer-eval

$(STATEDIR_BUILT):
	mkdir -p $(STATEFILES)
	touch $(STATEDIR_BUILT)

$(DOCKER_BUILT): $(STATEDIR_BUILT) Dockerfile build-files/* Makefile requirements.txt
	docker build -t $(DOCKER_TAG) .
	touch $(DOCKER_BUILT)

docker: $(DOCKER_BUILT)
