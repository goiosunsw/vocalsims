from .json_object import JSONObject



def frange(start, stop, step):
    x = start
    while x<stop:
        yield x
        x+=step

def json_expander(jsource,jrules,ignore_kw=None):
    props = {}
    for p,v in jrules.iter_tree():
        val=v
        gen=None
        try:
            idx = p.index("_range")
            print(idx)
        except ValueError:
            idx = -1
        if idx > -1:
            path = p[:idx+1]
            val = jrules[path]
            print(path)
            gen = frange(val['start'],val['end'],val['step'])            

        try:
            idx = p.index("_list")
            print(idx)
        except ValueError:
            idx = -1
        if idx > -1:
            path = p[:idx+1]
            val = jrules[path]
            print(path)
            gen = val['vals'].to_python()            
        ignore = False
        try:
            ignore = val[ignore_kw]
        except (KeyError, TypeError):
            pass
        if not ignore and gen:
            props[tuple(path[:-1])] = gen
        
    print(props)
    from itertools import product
    for x in product(*props.values()):
        jtarget = jsource.copy()
        for k,xx in zip(props.keys(),x):
            jtarget[k]= xx
        yield jtarget
    

if __name__ == '__main__':
    import sys
    with open(sys.argv[1]) as f:
        jj = JSONObject(f)
        #js = {'glottis':{'lf coefficients':{'rg':2}}}
        # js = jj['sequence']
        js = jj.pop('sequence')

    print(json.dumps(json_replacer(jj,js),indent=2))
#    for p,v in tree_path_iterate(jj):
#        try:
#            v2=js
#            for pp in p:
#                v2 = v2[pp]
#            print ("R: {} :: {} -> {}".format(str(p),v,str(v2))) 
#        except (KeyError,IndexError):
#            print("K: {} :: {}".format(str(p),v))