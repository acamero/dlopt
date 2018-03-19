# Deep learning architecture optimization

Metaheuristic procedures for recurrent neural network (RNN) optimization.


## Index

This respository contains several deliverables associated to the main topic (RNN optimization). The main structure is as follows:

* **architecture**: metaheuristic based RNN architecture optimization and backpropagation training.
* **data**: datasets used in publications.
* **etc**: analysis, useful scripts, and other stuff.
* **publications**: stand alone deliverables related to scientific publications. Please refer to [Publications](#publications).
* **weight**: metaheuristic based RNN architecture and weights optimization.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Each deliverable (sub project) contains a **requirements.txt** file listing the Python packages needed to tun the code.

### Installing

Clone the repository on your local machine, install the dependencies (you may want to use a virtual environment), and have fun!

### Building

To build (binary) ann application (e.g. optimizer.py) you could use **pyinstaller**. Remember to add the path to required libraries and any hidden import, for example:

```
pyinstaller optimizer.py -F --path ../env/lib/python3.5/ --hidden-import algorithms
```

## Publications

* Camero A., Toutouh, J., Stolfi, D.H., and Alba, E. Evolutionary Deep Learning for Car Park Occupancy Prediction in Smart Cities. To appear in Proc. of Learning and Intelligent OptimizatioN Conference (LION 12). 2008.


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.


## Authors

* [**Andrés Camero**](http://neo.lcc.uma.es/staff/acamero/) - *Initial work* - [dlopt](https://github.com/acamero/dlopt)

Please see the list of scientific publications for more information about people who has participated in this project or visit [NEO](http://neo.lcc.uma.es) research webpage.




## License

This project is licensed under the GNU GPL v3 license - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

This project has been partially funded by the Spanish MINECO and FEDER projects:
* [TIN2014-57341-R](http://moveon.lcc.uma.es)
* [TIN2016-81766-REDT](http://cirti.es)
* [TIN2017-88213-R](http://6city.lcc.uma.es)

