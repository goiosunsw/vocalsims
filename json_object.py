import json as jpy

class JSONObject(object):
    def __init__(self, json=None):
        self.max_repr_chars=160
        self._json={}
        if json is not None:
            try: 
                self._json = jpy.load(json)
                return
            except AttributeError:
                pass
            try:
                self._json = jpy.loads(json)
                return
            except TypeError:#jpy.JSONDecodeError:
                self._json = json

    def _normalise_key(self,key):
        try:
            if len(key) == 0:
                return key
        except TypeError:
            # key is integer
            return str(key)
        if key[0] == '/':
            key = key[1:]
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
                    return self.__class__(json=el)
            else:
                return el
        except TypeError:
            return el

    def __setitem__(self,key,value):
        key = self._normalise_key(key)
        el = self._json
        k2 = key[-1]
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
            for ii in range(len(el),key[-1]):
                el.append({})

    def __len__(self):
        return len(self._json)

    def items(self):
        try:
            for k, v in self._json.items():
                if isinstance(v, (dict,list,tuple)):
                    yield k, self.__class__(v)
                else:
                    yield k,v
        except AttributeError:
            for kk, vv in enumerate(self._json):
                yield kk, vv

    def read_file(self,filename):
        with open(filename,'r') as f:
            self._json = jpy.load(f)    

    def iter_tree(self):
        self._cur_path = []
        yield from self._iter_tree()

    def _iter_tree(self):
        cur_node = self[self._cur_path]
        cur_path = self._cur_path
        # print("-- "+str(cur_path))
        try:
            len(cur_node) == 0
        except TypeError:
            yield cur_path, cur_node
        else:    
            if isinstance(cur_node, self.__class__):
                for key, value in cur_node.items():
                    self._cur_path = cur_path + [key]
                    yield from self._iter_tree()
            else:
                yield cur_path, cur_node
            
                #self._cur_path = cur_path
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
        for k, v in self.iter_tree():
            fun(v)

    def visititems(self, fun):
        for k, v in self.iter_tree():
            fun(k,v)
    
    def __repr__(self):
        ret = str(self.__class__)+':\n'
        jsstr = str(self._json)
        if len(jsstr)>self.max_repr_chars:
            nch = self.max_repr_chars//2 - 2
            jsstr = jsstr[:nch]+' ... '+jsstr[-nch:]
        ret += jsstr
        return ret 
            
    def copy(self):
        newj = self.__class__()
        for k,v in self.iter_tree():
            newj[k] = v
        return newj

    def _get_raw_json_parent(self,key):
        key = self._normalise_key(key)
        el = self._json
        for kk in key[:-1]:
            el = el[kk] 
        return el

    def __delitem__(self,key):
        key = self._normalise_key(key)
        parent = self._get_raw_json_parent(key)
        del parent[key[-1]]

    def pop(self, key):
        key = self._normalise_key(key)
        parent = self._get_raw_json_parent(key)
        ret = parent[key[-1]]
        del parent[key[-1]]
        return self.__class__(ret)

    def to_python(self):
        return self._json

    def dumps(self, indent=None):
        return jpy.dumps(self._json, indent=indent)

    def dump(self,f):
        return jpy.dump(self._json,f)

        
            
            
        
        
        
        
        