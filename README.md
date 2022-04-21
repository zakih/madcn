# Multi-Agent Data Compression Network (MADCN): linear programming optimizations for data flow problems

Network consists of three types of agents which we represent as nodes:

  1. Nodes which transmit, receive, and compress data. We refer to these relay nodes and indicate them with the suffix 'r' in diagrams.
  2. Nodes which, in addition to transmitting, receiving and compressing, also produce data. We refer to these as data source nodes and indicate them with the suffix 'ds' in diagrams
  3. A solitary node which only receives data, thereby acting as a data sink in the network. We refer to this as the base node.

## Installation

1. Requires [glpk](https://www.gnu.org/software/glpk/) C tools. Instructions for [Linux](https://en.wikibooks.org/wiki/GLPK/Linux_OS), [MacOS](https://en.wikibooks.org/wiki/GLPK/Mac_OS_X)
2. Requries Python modules [PyGLPK](https://pypi.org/project/glpk/), [NetworkX](https://pypi.org/project/networkx/), [PyYAML](https://pypi.org/project/PyYAML/), and [Numpy](https://pypi.org/project/numpy/). Can install using pip: `pip3 install -r requirements.txt`

## How to use

1. Specify options in `/config.yaml` and run `python3 madcn.py` with flags (discussed below)
2. Use CLI functionality with following options.

    * Pick which problem(s) to solve, must select at least one
      * `--minenergy` to solve minimum energy problem
      * `--minuptime` to solve minimum uptime problem
      * `--maxdata` to solve max data problem

    * Generate network randomly or specify data file. Defaults to using sample graph data file path given by `graph_data_file` in `/config.yaml` if no options passed.
      * `--graph` path to graph data JSON file, e.g. `data/sample_graph_data.json`. Overrides `graph_data_file` in `/config.yaml`
      * `--random` use this to generate a random network with parameters given in the `config.yaml`

    * Simulation outputs
      * `--viz` use this to draw network diagrams for original graph and each solution data flow
      * `--v` use this for verbose output in terminal
      * `--sim` use this to specify a simulation tag. Overrides `sim_name` in `/config.yaml`

    * Specify limits for max data problem (problem 3)
      * `--E` use this to specify maximum energy for data flow, this will override option `max_energy` in `/config.yaml`
      * `--T` use this to specify maximum makespan time for data flow, this will override option `max_makespan` in `/config.yaml`

3. Options in `/config.yaml`

    * Specify path to directory where outputs should be saved in `output_dir`
    * Specify energy and time cost rates for edges in network under `cost_rates`. For example, energy rates can be joules/bits, and time rates can be seconds/bits

4. Example runs

    * Get max data flow to base station in a random network while ouptutting plots of solution data flows and maximum energy of 3000:
      * `python3 madcn.py --random --maxdata --viz --E 3000`
    * Get max data and min energy for a network specified in a file `path/to/my/graph.json`:
      * `python3 madcn.py --maxdata --minenergy --graph path/to/my/graph.json`
    * Get max data and min uptime solution graphs in a random network with default parameters with the simulation name `test`:
      * `python3 madcn.py --random --maxdata --minuptime --sim test`

5. Outputs

    * Results for problems 1, 2, and 3 are appended to `result_min_energy_problem.csv`, `result_min_uptime_problem.csv`, `result_max_data_problem.csv` respectively in the output directory. One row is added for each simulation.
    * The data flow is also output as a plan, consisting of data flow paths from each source to base station in `data_flow_min_energy_<sim_name>.json`, `data_flow_min_uptime_<sim_name>.json`, `data_flow_max_data_<sim_name>.json` for problems 1, 2, and 3 respectively. These files also list:
      * total data volume transferred to base
      * total energy required for the data flow
      * total uptime required for the data flow
      * makespan of the data flow

## Copyright

  ```
  Copyright (c) 2022 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
  following conditions are met:

  • Redistributions of source code must retain the above copyright notice, this list of conditions and the following
  disclaimer.

  • Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
  disclaimer in the documentation and/or other materials provided with the distribution.

  • Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its
  contributors may be used to endorse or promote products derived from this software without specific prior written
  permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  ```
