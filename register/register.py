from collections import abc

def is_ele_with_type(seq_data, ele_type, seq_type=None):
    """
    :desc judge if the data with seq_type and all items are ele_type
    :param seq_data: sequence data
    :param ele_type: sequence element type
    :param seq_type: sequence type
    :return: True if meet condition else False
    """
    if seq_type is None:
        seq_exp_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        seq_exp_type = ele_type

    if not isinstance(seq_data, seq_exp_type):
        return False

    for ele in seq_data:
        if not isinstance(ele, ele_type):
            return False
    return True


class Register(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def __len__(self):
        return len(self._module_dict)

    def __repr__(self):
        fmt_str = "Class name: {}, Register name: {}, Module dict: {}".format(self.__class__.__name__,
                                                                              self._name,
                                                                              self._module_dict)
        return fmt_str

    def get(self, name):
        return self._module_dict.get(name, None) # 返回指定键的值

    def __contains__(self, item):
        """
        :desc if contains registered obj
        :param item: obj name
        :return: True if contains else False
        """
        return self.get(item) is not None

    def _do_register(self, object_object, object_name, override=False):
        """
        :desc register object_name in module_dict
        :param object_name: object name
        :param object_object: object of the object
        :param override: override object_instance if object_name exists
        :return:
        """
        if type(object_name) is str:
            object_name = [object_name]
        else:
            assert is_ele_with_type(object_name, str), "object_name should be sequence of string or string"

        for obj_name in object_name:
            if not override and obj_name in self._module_dict:
                raise ValueError("{} has already registered in Register,"
                                 "keys: {}".format(obj_name, list(self._module_dict.keys())))
            self._module_dict[obj_name] = object_object

    def deregister(self, object_name):
        """
        :desc remove object in module_dict
        :param object_name: object name
        :return:
        """
        if object_name in self._module_dict:
            self._module_dict.pop(object_name)

    def _check_register_params(self, object_name, override):
        if (object_name is not None) and (not isinstance(object_name, str)):
            raise ValueError("object_name should be string")

        if not isinstance(override, bool):
            raise ValueError("override should be bool")

    def register_module(self, module_object=None, module_name=None, override=False):
        return self.register_class(module_object, module_name, override)

    def register_class(self, class_object=None, class_name=None, override=False):
        """
        :desc register class
        :param class_object: class object
        :param class_name: class name
        :param override: override class_object if class_name exists
        :return:

        :example
            LOSSES = Register("LOSSES")
        #1.
            @LOSSES.register_class()
            class XX_LOSS():
                ...
        #2.
            class XX_LOSS():
                ...

            LOSS_REGIS.register_class(XX_LOSS)
        """
        self._check_register_params(class_name, override)

        if class_object is not None:
            # register through invoking
            class_name = class_object.__name__ if class_name is None else class_name
            self._do_register(class_object, object_name=class_name, override=override)
            return class_object
        else:
            # register through decoration
            def _register(cls):
                class_name = cls.__name__
                self._do_register(cls, object_name=class_name, override=override)
                return cls

            return _register

    def register_class_func(self, class_func_object=None, class_func_name=None, override=False):
        self._check_register_params(class_func_name, override)
        # TODO register class func can't instantiate from cla_config

        if class_func_object is not None:
            # register through invoking
            class_func_name = class_func_object.__name__ if class_func_name is None else class_func_name
            self._do_register(class_func_object, class_func_name, override)
            return class_func_object
        else:
            # register through decoration
            def _register(func):
                class_func_name = func.__name__
                self._do_register(func, class_func_name, override)
                return func
            return _register

    def register_function(self, function_object=None, function_name=None, override=False):
        self._check_register_params(function_name, override)

        if function_object is not None:
            # register through invoking
            function_name = function_object.__name__ if function_name is None else function_name
            self._do_register(function_object, function_name, override)
            return function_object
        else:
            # register through decoration
            def _register(func):
                function_name = func.__name__
                self._do_register(func, function_name, override)
                return func
            return _register


