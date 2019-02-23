#! /bin/bash
set -e -x
export package_version="${1:-$(date +'%Y.%-m.%-d.%-H.%-M.%-S')}"
echo "Building manylinux1 wheels, version '${package_version}'"


### NOTE: This script is designed to be written in the standard pypa/manylinux1 docker container.
### That container is an intentionally old CentOS 5 container, with a bunch of built-in goodies to
### build python wheels that can be used on any reasonably modern flavor of linux.  To use this, run
### a command like
###
###     docker run -i -t \
###         -v ${HOME}/.pypirc:/root/.pypirc:ro \
###         -v /full/path/to/code:/code \
###         -v /full/path/to/this/script/build_maylinux_wheels.sh:/build_manylinux_wheels.sh \
###         quay.io/pypa/manylinux1_x86_64 /build_manylinux_wheels.sh
###
### Also be aware that several lines of this script may be specialized for this particular code; see
### the comments above each section to decide if that is true.
###
### You can also add an argument at the end of that line containing a specialized version string.
### This defaults to the date and time in '%Y.%m.%d.%H.%M.%S' format, which is made available to the
### setup.py script as the `datetime` environment variable for incorporation into the pip version
### string.  Passing this explicitly is nice because it allows other builds to have the same
### version, which means pypi will display them on the same screen, and pip will treat them equally.




/bin/rm -rf /wheelhouse
mkdir -p /wheelhouse

PYBINS=()
for PYBIN in /opt/python/*/bin; do
    if [[ "${PYBIN}" == "/opt/python/cp26-cp26m/bin"
          || "${PYBIN}" == "/opt/python/cp26-cp26mu/bin"
          || "${PYBIN}" == "/opt/python/cp27-cp27m/bin"
          || "${PYBIN}" == "/opt/python/cp27-cp27mu/bin"
          || "${PYBIN}" == "/opt/python/cp33-cp33m/bin" ]]; then
        continue
    fi
    PYBINS+=("${PYBIN}")
    LAST_PYBIN="${PYBIN}"
done

# Compile wheels
for PYBIN in "${PYBINS[@]}"; do
    ### NOTE: The path to the requirements file is specialized for spinsfast
    PYTHON_VERSION=$("${PYBIN}/python" -c 'import sys; print("{0}{1}".format(sys.version_info.major, sys.version_info.minor))')
    if [ "${PYTHON_VERSION}" -gt "36" ]; then
        requirements_build_txt="requirements-build-115.txt"
    else
        requirements_build_txt="requirements-build.txt"
    fi
    "${PYBIN}/pip" install --upgrade pip wheel==0.31.1 auditwheel
    "${PYBIN}/pip" install -r "/code/${requirements_build_txt}"
    "${PYBIN}/pip" wheel /code/ -r "/code/${requirements_build_txt}" -w /wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in /wheelhouse/numpy_quaternion*manylinux*.whl; do
    auditwheel repair "$whl" -w /wheelhouse/
done

for PYBIN in "${PYBINS[@]}"; do
    # Install packages and test ability to import and run simple command
    "${PYBIN}/pip" install numpy-quaternion --no-index -f /wheelhouse
    (cd "$HOME"; "${PYBIN}/python" -c 'import numpy as np; import quaternion; print(quaternion.__version__); print("quaternion.z = {0}".format(quaternion.z))')
done


# Upload to pypi
"${LAST_PYBIN}"/pip install twine
"${LAST_PYBIN}"/twine upload /wheelhouse/numpy_quaternion*manylinux*.whl
