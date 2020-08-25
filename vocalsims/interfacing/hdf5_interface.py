import h5py


class Hdf5Interface(object):
    def __init__(self, file=None, path=None, mode='a'):
        self._file = None
        self.set_path(file=file,path=path,mode='a')
        self._unnamed_count = 0
    
    def set_path(self, file, path=None, mode='a'):
        with h5py.File(file, mode=mode) as f:
            g = f.require_group(path)
        self._file = file
        self._path = path

    def _get_first_free_unnamed(self):
        with h5py.File(self._file, mode="r") as f:
            while True:
                name = "{}Unnamed{}".format(self.path,self._unnamed_count)
                try:
                    f[name]
                    self._unnamed_count += 1
                except KeyError:
                    return self._unnamed_count
                
    def write_dataset(self, data, name=None):
        if name is None:
            self._get_first_free_unnamed()
            name = "Unnamed{}".format(self._unnamed_count)
        with h5py.File(self._file, mode="a") as f:
            gg = f.create_dataset(self._path+'/'+name, data=data)

    def write_attrs(self, attrs, group=""):
        if group is None:
            self._get_first_free_unnamed()
            group = "Unnamed{}".format(self._unnamed_count)
        with h5py.File(self._file, mode="a") as f:
            gg = f.require_group(self._path+'/'+group)
            for k,v in attrs.items():
                gg.attrs[k] = v
                

        

        
        