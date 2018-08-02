hellomake: preprocessing/augmentation_preprocessing.cpp preprocessing/graph_construction.cpp \
           preprocessing/graph_construction.h preprocessing/parameters.h \
           preprocessing/wrapper_interface.cpp preprocessing/py_graph_construction.pyx
	python setup.py build_ext -i --force
