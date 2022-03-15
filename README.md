VENUS 2
=======

# Description
--------------

Venus is a complete verification tool for Relu-based feed-forward neural
networks.  Venus implements a MILP-based verification method whereby it
leverages dependency relations between the ReLU nodes to reduce the search
space that needs to be considered during branch-and-bound. The dependency
relations are exploited via callback cuts and via a branching method that
divides the verification problem into a set of sub-problems whose MILP
formulations require fewer integrality constraints. 

# Requirements 
---------------

* Python 3.7 or higher 
* Gurobi 9.1 or higher. 

# Installation
---------------

## Install Gurobi

```sh
wget https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz
tar xvfz gurobi9.5.1_linux64.tar.gz
```

### Add the following to the .bashrc file:
```sh
export GUROBI_HOME="Current_directory/gurobi951/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

### Retrieve a Gurobi License

To run Gurobi one needs to obtain license from [here](https://www.gurobi.com/documentation/9.5/quickstart_linux/retrieving_and_setting_up_.html#section:RetrieveLicense).


## Install Venus
```
pipenv install
pipenv shell
```

# Usage
-----------

```sh
python3 . 
    --net <path to network in onnx format> 
    --spec <path to spec in vnnlib format or path to folder with vnnlib specs>
    --timeout <timeout in seconds>
```


# Contributors
--------------

* Panagiotis Kouvaros (lead contact) - p.kouvaros@imperial.ac.uk

* Alessio Lomuscio - a.lomuscio@imperial.ac.uk


# License and Copyright
---------------------

Licensed under the [BSD-2-Clause](https://opensource.org/licenses/BSD-2-Clause)

