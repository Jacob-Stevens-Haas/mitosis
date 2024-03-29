from logging import getLogger


def run(**kwargs):
    getLogger(__name__).info("This is run every time")
    getLogger(__name__).debug("This is run in debug mode only")
    return {"main": 0}


lookup_dict = {"foo": {"test": 2}}


class MockExp:
    class MockExpInner:
        lookup_dict: dict[str, dict[str, int]] = lookup_dict

        @staticmethod
        def run():
            return {"main": 0}
