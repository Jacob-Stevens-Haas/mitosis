from collections import defaultdict

data_config = {"length": {"test": 5}}

meth_config = {"metric": {"test": "len"}}

# lookup any parameter, any variant: always none
lookup_default = defaultdict(lambda: defaultdict(lambda: None))
