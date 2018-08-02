# -*- coding: utf-8 -*-
from collections import defaultdict
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import subprocess
import numpy
import sys


def pkgconfig(flag):
    p = subprocess.Popen(['pkg-config', flag] +
                         pcl_libs, stdout=subprocess.PIPE)
    stdout, _ = p.communicate()
    return stdout.decode().split()


# Try to find PCL. XXX we should only do this when trying to build or install.
PCL_SUPPORTED = ["-1.7", ""]    # in order of preference

for pcl_version in PCL_SUPPORTED:
    if subprocess.call(['pkg-config', 'pcl_common%s' % pcl_version]) == 0:
        break
    else:
        print "%s: error: cannot find PCL, tried" % sys.argv[0]
        for version in PCL_SUPPORTED:
            print '    pkg-config pcl_common%s' % version
        sys.exit(1)

# Find build/link options for PCL using pkg-config.
pcl_libs = ["common", "features", "filters", "io", "kdtree", "octree",
            "registration", "sample_consensus", "search", "segmentation",
            "surface", "tracking", "visualization"]
pcl_libs = ["pcl_%s%s" % (lib, pcl_version) for lib in pcl_libs]

ext_args = defaultdict(list)
ext_args['include_dirs'].append(numpy.get_include())

for flag in pkgconfig('--cflags-only-I'):
    ext_args['include_dirs'].append(flag[2:])

ext_args['include_dirs'].append('/usr/include/vtk-6.2')
ext_args['library_dirs'].append('/usr/lib')


for flag in pkgconfig('--cflags-only-other'):
    if flag.startswith('-D'):
        macro, value = flag[2:].split('=', 1)
        ext_args['define_macros'].append((macro, value))
    else:
        ext_args['extra_compile_args'].append(flag)

for flag in pkgconfig('--libs-only-l'):
    if flag == "-lflann_cpp-gd":
        continue
    ext_args['libraries'].append(flag[2:])

for flag in pkgconfig('--libs-only-L'):
    ext_args['library_dirs'].append(flag[2:])

for flag in pkgconfig('--libs-only-other'):
    ext_args['extra_link_args'].append(flag)


ext_args['define_macros'].append(
    ("EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET", "1"))

ext_args['extra_compile_args'].append("-std=c++11")
ext_args['extra_link_args'].append("-std=c++11")
ext_args['extra_link_args'].append('-lboost_system')

for k, v in ext_args.iteritems():
    print k, ":", v
print "\n\n"

setup(
    name='Graph Structure Estimation',
    ext_modules=cythonize(Extension(
        "py_graph_construction",
        ["preprocessing/py_graph_construction.pyx"],
        language="c++",
        **ext_args))
)
