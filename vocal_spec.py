from json_object import JSONObject
from datetime import datetime

__parameter_version__ = "20191128"


class VocalSpec(JSONObject):
    """
    Class specifyin vocal simulation parameters
    Reads from JSON
    """   
    def __init__(self,json=None):
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        super().__init__(json=json)

    def set_aliases(self, alias_dict):
        self._alias_dict = alias_dict
        for k, v in alias_dict.items():
            self.__setattr__(k,self[v]) 

    def __getattr__(self,key):
        try:
            return(self[key])
        except KeyError:
            super().__getattr__(key)
        