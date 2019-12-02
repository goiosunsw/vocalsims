import json

class JSONObject(object):
    def __init__(self, json=None):
        self.max_repr_chars=160
        if json is not None:
            self._json = json
    def __getattr__(self, key):
        key = key.split('/')
        return self.__getitem__(key)
    def _normalise_key(self,key):
        if isinstance(key,int):
            key = str(key)
        if isinstance(key,str):
            key = key.split('/')
        return key
        
    def __getitem__(self,key):
        key = self._normalise_key(key)
        el = self._json
        for kk in key:
            try:
                kk = int(kk)
            except ValueError:
                pass
            el = el[kk]
        try:
            if len(el)>0:
                if isinstance(el,str):
                    return el
                else:
                    return JSONObject(json=el)
            else:
                return el
        except TypeError:
            return el
    def __setitem__(self,key,value):
        key = self._normalise_key(key)
        el = self._json
        for k1, k2 in zip(key[:-1],key[1:]):
            try:
                k1 = int(k1)
            except ValueError:
                pass
            try:
                k2 = int(k2)
            except ValueError:
                pass
            try:
                el = el[k1]
            except KeyError:
                if isinstance(k2,int):
                    el[k1] = [{}]*(k2+1)
                    el = el[k1]
                else:
                    el[k1] = {}
                    el = el[k1]
            except IndexError:
                for ii in range(len(el),k1):
                    el.append(None)
                el.append({})
                el = el[k1]
        try:       
            el[k2] = value
        except IndexError:
            for ii in range(len(el),path[-1]):
                el.append({})
    def __len__(self):
        return len(self._json)
    def items(self):
        try:
            yield from self._json.items()
        except AttributeError:
            for kk, vv in enumerate(self._json):
                yield kk, vv
    def read_file(self,filename):
        with open(filename,'r') as f:
            self._json = json.load(f)    
    def iter_tree(self):
        self._cur_path = []
        yield from self._iter_tree()
    def _iter_tree(self):
        cur_node = self[self._cur_path]
        cur_path = self._cur_path
        try:
            len(cur_node) == 0
        except TypeError:
            yield self._cur_path, cur_node
        
        if isinstance(cur_node, JSONObject):
            for key, value in cur_node.items():
                self._cur_path = cur_path + [key]
                yield from self._iter_tree()
#         if isinstance(cur_node, dict):
#             if len(cur_node) == 0:
#                 yield cur_path, cur_node
#             for key, value in cur_node.items():
#                 self._cur_path = cur_path + [key]
#                 yield from self._iter_tree()
#         if isinstance(cur_node, (list,tuple)):
#             for key, value in enumerate(cur_node):
#                 self._cur_path = cur_path + [key]
#                 yield from self._iter_tree()
    def visit(self,fun):
        ret = []
        for k, v in self.iter_tree():
            ret.append(fun(v))
        return ret
    def __repr__(self):
        ret = 'JSONObject:\n'
        jsstr = str(self._json)
        if len(jsstr)>self.max_repr_chars:
            nch = self.max_repr_chars//2 - 2
            jsstr = jsstr[:nch]+' ... '+jsstr[-nch:]
        ret += jsstr
        return ret 
            
            
        
        
        
        
        