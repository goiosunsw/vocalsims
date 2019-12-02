import json

path = []

def rec_dec(f):
    path = []
    def wrapper(*argv):
        return(f(*argv))
    return wrapper


# @rec_dec
# def tree_path_iterate(dictionary):
#     print(path)
#     for key, value in dictionary.items():
#         if isinstance(value, dict):
#             path.append(key)
#             if len(value)>0:
#                 yield from tree_path_iterate(value)
#             else:
#                 yield path+[key],value
#             path.pop()
#         elif isinstance(value, list):
#             path.append(key)
#             path.append(0)
#             for el in value:
#                 yield from tree_path_iterate(el)
#                 path[-1]+=1
#             path.pop()
#             path.pop()
#         else:
#             yield path+[key],value


def tree_path_iterate(dictionary):
    path=[]
    def _tree_path_iter(dictionary):
        if len(dictionary)==0:
            yield path, dictionary
        for key, value in dictionary.items():
            if isinstance(value, dict):
                path.append(key)
                if len(value)>0:
                    yield from _tree_path_iter(value)
                else:
                    yield path+[key],value
                path.pop()
            elif isinstance(value, list):
                path.append(key)
                path.append(0)
                for el in value:
                    yield from _tree_path_iter(el)
                    path[-1]+=1
                path.pop()
                path.pop()
            else:
                yield path+[key],value
    yield from _tree_path_iter(dictionary)

def set_json_path(js, path, val):
    el = js
    for p1, p2 in zip(path[:-1],path[1:]):
        try:
            el = el[p1]
        except KeyError:
            if isinstance(p2,int):
                el[p1] = [{}]*(p2+1)
                el = el[p1]
            else:
                el[p1] = {}
                el = el[p1]
        except IndexError:
            for ii in range(len(el),p1):
                el.append(None)
            el.append({})
            el = el[p1]
                
    el[path[-1]] = val
        

def json_pop_path(jsource,path):
    jtarget = {}
    jpop = {}
    for p,v in tree_path_iterate(jsource):
        if all([p1==p2 for p1,p2 in zip(p,path)]):
            set_json_path(jpop,p,v)
        else:
            set_json_path(jtarget,p,v)

    return jtarget, jpop
    

def json_replacer(j1,j2):
    jr = {}
    for p,v in tree_path_iterate(j1):
        try:
            v2=j2
            for pp in p:
                v2 = v2[pp]
            vr = v2 
            print('REPLACED')
        except (KeyError,IndexError):
            vr = v
        set_json_path(jr,p,vr)

            
    return jr

def json_deepcopy(jsource):
    jtarget = {}
    for p,v in tree_path_iterate(jsource):
        set_json_path(jtarget,p,v)

    return jtarget

def multi_json_generator(j1,j2):
    jr = {}
    for p,v in tree_path_iterate(j1):
        try:
            v2=j2
            for pp in p:
                v2 = v2[pp]
            vr = v2 
            print('REPLACED')
        except (KeyError,IndexError):
            vr = v
        set_json_path(jr,p,vr)

            
    return jr

def frange(start, stop, step):
    x = start
    while x<stop:
        yield x
        x+=step

def json_expander(jsource,jrules):
    props = {}
    for p,v in tree_path_iterate(jrules):
        val = jrules
        for ip,pp in enumerate(p):
            val = val[pp]
            if pp == "_range":
                props[tuple(p[:ip])] = frange(val['start'],val['end'],val['step'])
                break
    print(props)
    from itertools import product
    for x in product(*props.values()):
        jtarget = json_deepcopy(jsource)
        for k,xx in zip(props.keys(),x):
            set_json_path(jtarget, k, xx)
        yield jtarget
    
    
    

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f:
        jj = json.load(f)
        #js = {'glottis':{'lf coefficients':{'rg':2}}}
        js = jj['sequence']

    print(json.dumps(json_replacer(jj,js),indent=2))
#    for p,v in tree_path_iterate(jj):
#        try:
#            v2=js
#            for pp in p:
#                v2 = v2[pp]
#            print ("R: {} :: {} -> {}".format(str(p),v,str(v2))) 
#        except (KeyError,IndexError):
#            print("K: {} :: {}".format(str(p),v))