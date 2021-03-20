import os
import os.path as osp
import sys
from glob import glob
from setuptools import dist, setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
    EXT_TYPE = "pytorch"
except ModuleNotFoundError:
    from Cython.Distutils import build_ext as BuildExtension
    print("Skip building ext ops due to the absence of torch.")


def get_requirements(filename="requirements.txt"):
    assert osp.exists(filename), f"{filename} not exists"
    with open(filename, "r") as f:
        content = f.read()
    lines = content.split("\n")
    requirements_list = list(
        filter(lambda x: x != "" and not x.startswith("#"), lines))
    return requirements_list


def get_version():
    version_file = osp.join("gorilla3d", "version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def get_sources(module, surfix="*.c*"):
    src_dir = osp.join(*module.split("."), "src")
    cuda_dir = osp.join(src_dir, "cuda")
    cpu_dir = osp.join(src_dir, "cpu")
    return glob(osp.join(src_dir, surfix)) + \
           glob(osp.join(cuda_dir, surfix)) + \
           glob(osp.join(cpu_dir, surfix))


def get_include_dir(module):
    include_dir = osp.join(*module.split("."), "include")
    if osp.exists(include_dir):
        return [osp.abspath(include_dir)]
    else:
        return []


def make_extension(name, module):
    if not torch.cuda.is_available(): return
    extersion = CUDAExtension
    return extersion(name=".".join([module, name]),
                     sources=get_sources(module),
                     include_dirs=get_include_dir(module),
                     extra_compile_args={
                         "cxx": ["-g"],
                         "nvcc": [
                             "-D__CUDA_NO_HALF_OPERATORS__",
                             "-D__CUDA_NO_HALF_CONVERSIONS__",
                             "-D__CUDA_NO_HALF2_OPERATORS__",
                         ],
                     },
                     define_macros=[("WITH_CUDA", None)])


def get_extensions():
    extensions = []
    if torch.cuda.is_available():
        extensions = [
            make_extension(name="compiling_info",
                           module="gorilla3d.ops.utils"),
            make_extension(name="ball_query_ext",
                           module="gorilla3d.ops.ball_query"),
            make_extension(name="group_points_ext",
                           module="gorilla3d.ops.group_points"),
            make_extension(name="interpolate_ext",
                           module="gorilla3d.ops.interpolate"),
            make_extension(name="furthest_point_sample_ext",
                           module="gorilla3d.ops.furthest_point_sample"),
            make_extension(name="gather_points_ext",
                           module="gorilla3d.ops.gather_points"),
            make_extension(name="chamfer",
                           module="gorilla3d.ops.chamfer_distance"),
            make_extension(name="sparse_interpolate_ext",
                           module="gorilla3d.ops.sparse_interpolate"),
        ]
    return extensions


if __name__ == "__main__":
    setup(name="gorilla3d",
          version=get_version(),
          author="Gorilla Authors",
          author_email="mszhihaoliang@mail.scut.edu.cn",
          description="3D vision library for Gorilla-Lab using PyTorch",
          long_description=open("README.md").read(),
          license="MIT",
          install_requires=get_requirements(),
          packages=find_packages(exclude=["tests"]),
          ext_modules=get_extensions(),
          cmdclass={"build_ext": BuildExtension},
          zip_safe=False)
