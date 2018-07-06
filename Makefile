hellomake: preprocessing/augmentation_preprocessing.cpp preprocessing/features_computation.cpp \
           preprocessing/graph_structure.cpp preprocessing/parameters.h \
           preprocessing/wrapper_interface.cpp preprocessing/graph_extraction.pyx
	python setup.py build_ext -i --force
