# DLOPT: Deep Learning Optimization

Python library for artificial neural network (NN) optimization.


## Index

This respository contains several deliverables associated to the main topic (NN optimization). The main structure is as follows:

* **data**: datasets used in publications.
* **dlopt**: main library
* **docs**: miscellaneous documents
* **etc**: analysis, useful scripts, and other stuff.
* **examples**: a quick guide on using DLOPT
* **publications**: stand alone deliverables related to scientific publications. Please refer to [Publications](#publications).


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

The main folder contains a **requirements.txt** file listing the Python packages needed to tun the code.

### Installing

Clone the repository on your local machine, install the dependencies (you may want to use a virtual environment), and have fun!

### Building

To build a binary application you could use **pyinstaller**. Remember to add the path to required libraries and any hidden import, for example:

```
pyinstaller optimizer.py -F --path ../env/lib/python3.5/ --hidden-import algorithms
```


## How to cite DLOPT

We encourage authors of scientific papers to cite:

* Camero, A., Toutouh, J., and Alba, E. DLOPT: Deep Learning Optimization Library. arXiv preprint arXiv:1807.03523 (july 2018)



## Related Publications

* Camero, A., Toutouh, J., Stolfi, D.H., and Alba, E. Evolutionary Deep Learning for Car Park Occupancy Prediction in Smart Cities. To appear in Proc. of Learning and Intelligent OptimizatioN Conference (LION 12). 2018.
* Camero, A., Toutouh, J., and Alba, E. Low-cost recurrent neural network expected performance evaluation. arXiv preprint arXiv:1805.07159 (may 2018)
* Camero, A., Toutouh, J., and Alba, E. DLOPT: Deep Learning Optimization Library. arXiv preprint arXiv:1807.03523 (july 2018)
* Camero, A., Toutouh, J., and Alba, E. Comparing Deep Recurrent Networks Based on the MAE Random Sampling, a First Approach. To appear in Conference of the Spanish Association for Artificial Intelligence, CAEPIA, 2018.


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.


## Authors

* [**Andrés Camero**](http://neo.lcc.uma.es/staff/acamero/) - *Initial work* - [dlopt](https://github.com/acamero/dlopt)

Please see the list of scientific publications for more information about people who has participated in this project or visit [NEO](http://neo.lcc.uma.es) research webpage.


## License

This project is licensed under the GNU GPL v3 license - see the [LICENSE.md](LICENSE.md) file for details. 


## Acknowledgments

This research was partially funded by Ministerio de Economı́a, Industria y Competitividad, Gobierno de España, and European
Regional Development Fund grant numbers:

* [TIN2014-57341-R](http://moveon.lcc.uma.es)
* [TIN2016-81766-REDT](http://cirti.es)
* [TIN2017-88213-R](http://6city.lcc.uma.es)

