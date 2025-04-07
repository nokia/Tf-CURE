.PHONY: clean virtualenv test install

PYTHON_PATH := $(shell which python3)

test:
		coverage run

clean:
		find . -name '*.py[co]' -delete

virtualenv:
		virtualenv --python=/usr/bin/python3.11 --prompt="|> tf_cure <|" env
		env/bin/pip install --upgrade pip
		grep -v '^\-e' requirements.txt | cut -d = -f 1 | xargs -n1 env/bin/pip3 install -U
		#pip3 freeze | sed -ne 's/==.*//p' | xargs env/bin/pip3 install -U
		@echo
		@echo "Virtualenv created, use 'source env/bin/activate' to use it"
		@echo

install:
		sudo apt-get -y install python-virtualenv python3-dev
