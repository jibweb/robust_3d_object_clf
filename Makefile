hellomake: preprocessing/mesh_graph_construction.h preprocessing/mesh_graph_construction.cpp \
		   preprocessing/point_cloud_graph_construction.h preprocessing/point_cloud_graph_construction.cpp \
           preprocessing/augmentation_preprocessing.cpp preprocessing/parameters.h \
           preprocessing/wrapper_interface.cpp preprocessing/py_graph_construction.pyx
	python setup.py build_ext -i --force
